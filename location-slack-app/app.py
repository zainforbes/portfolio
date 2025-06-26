import os
import json
import math
import requests
from dotenv import load_dotenv
import openrouteservice
from openrouteservice.exceptions import ApiError
import folium
import openai
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Load environment variables
load_dotenv()

# OpenRouteService client setup
ors_api_key = os.getenv("ors_api_key")
if not ors_api_key:
    raise ValueError("Please set ors_api_key in your .env file.")
ors_client = openrouteservice.Client(key=ors_api_key)

# OpenAI setup
oai_key = os.getenv("open_api_key")
if not oai_key:
    raise ValueError("Please set open_api_key in your .env file.")
openai.api_key = oai_key

# Slack credentials
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
if not SLACK_BOT_TOKEN or not SLACK_SIGNING_SECRET or not SLACK_APP_TOKEN:
    raise ValueError("Please set SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, and SLACK_APP_TOKEN in your .env file.")

# Initialize Bolt app
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)

# Helper: Generate insight using OpenAI
def generate_insight(latitude, longitude, minutes, area_m2=None):
    area_km2 = area_m2 / 1e6 if area_m2 else 0
    prompt = f"""
You are a senior GIS data scientist with expertise in market and site‐selection analysis, especially for Out‐Of‐Home media and retail catchments in South Africa.

Given:
• Coordinates: ({latitude}, {longitude})
• Drive‐time: {minutes} minutes (≈ {area_km2:.2f} km²)

Produce a concise insight report with these sections, using bullet points and quantifiable estimates where possible:

1. **Location Identity**  
2. **Regional Context & Access**  
3. **Catchment Demographics**  
4. **Land Use & Amenities**  
5. **Footfall & Activation Opportunities**  
6. **Market Potential & Use Cases**  
7. **Key Observations**  

Keep it under ~200 words and finish with one sentence suggesting the best next analytical step.
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful GIS assistant."},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

# --- Slash Command: /isochrone ---
@app.command("/isochrone")
def handle_isochrone(ack, body, client):
    ack("🛰️ Generating your isochrone...")
    try:
        text = (body.get("text") or "").strip()
        parts = text.split()
        if len(parts) < 2:
            raise ValueError("Invalid format. Use `/isochrone lat,lon minutes`.")
        coords, mins_str = parts[0], parts[1]
        latitude, longitude = map(float, coords.split(","))
        minutes = int(mins_str)

        # Generate isochrone via ORS with area attribute
        try:
            geojson = ors_client.isochrones(
                locations=[[longitude, latitude]],
                profile="driving-car",
                range=[minutes * 60],
                attributes=["area"]
            )
        except ApiError as api_err:
            msg = api_err.error.get('message', str(api_err)) if hasattr(api_err, 'error') else str(api_err)
            client.chat_postEphemeral(
                channel=body["channel_id"],
                user=body["user_id"],
                text=f"⚠️ Could not build isochrone: {msg}. Try a smaller drive-time or check the coordinates."
            )
            return

        props = geojson["features"][0].get("properties", {})
        area_m2 = props.get("area")
        out_dir = "isochrones"
        os.makedirs(out_dir, exist_ok=True)
        geo_path = os.path.join(out_dir, f"isochrone_{latitude}_{longitude}_{minutes}.geojson")
        with open(geo_path, "w") as f:
            json.dump(geojson, f)

        # Create and save HTML map
        m = folium.Map(location=[latitude, longitude], zoom_start=13)
        folium.Marker(location=[latitude, longitude], icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        folium.GeoJson(geojson, name="isochrone").add_to(m)
        map_path = os.path.join(out_dir, f"isochrone_{latitude}_{longitude}_{minutes}.html")
        m.save(map_path)

        # Upload files to Slack
        #client.files_upload_v2(channel=body["channel_id"], file=geo_path, title=f"{minutes}-min isochrone (GeoJSON)", initial_comment=f"🚗 Generated {minutes}-min isochrone around {latitude},{longitude}.")
        #client.files_upload_v2(channel=body["channel_id"], file=map_path, title=f"{minutes}-min isochrone (Map)", initial_comment="🗺️ Interactive Folium map. Download to view.")

        # Generate & post insight summary
        insight = generate_insight(latitude, longitude, minutes, area_m2)
        client.chat_postMessage(channel=body["channel_id"], text=f"💡 Insight Summary:\n{insight}")

    except Exception as e:
        client.chat_postEphemeral(channel=body.get("channel_id"), user=body.get("user_id"), text=f"❌ Error: {e}")

if __name__ == "__main__":
    print("⚡️ IsoChrone-Getter is now running in Socket Mode…")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
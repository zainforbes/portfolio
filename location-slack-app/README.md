# Portfolio: Slack Bot for Location Intelligence

A Python Slack Bolt app that provides geospatial insights directly in Slack via slash commands. Currently supported:

* `/isochrone`: Generate drive-time isochrones, save GeoJSON & interactive map, and post AI-driven insights.

## Prerequisites

* Python 3.8+
* Slack App with Slash Commands configured and Socket Mode enabled
* Overpass API access (public endpoint)

## Installation

1. Clone this repo:

   ```bash
   git clone https://github.com/zainforbes/portfolio.git
   cd portfolio
   ```

2. Create a virtual environment & install:

   ```bash
   python -m venv env
   source env/bin/activate   # Mac/Linux
   env\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

3. Copy & populate environment variables:

   ```bash
   cp .env.example .env
   # then open .env and add your API keys + Slack tokens
   ```

## Usage

Run the app:

```bash
python app.py
```

Your bot will connect via Socket Mode. In any Slack channel or DM, type:

```bash
/isochrone lat,lon minutes
```

## Commands

### `/isochrone lat,lon minutes`

* Generates a drive-time isochrone (GeoJSON + Folium map).
* AI-driven summary insight from GPT-4o-mini.

## Extending

* Add more slash commands by registering in your Slack App settings and adding handlers in `app.py`.
* Use the helper functions (`generate_insight`) as templates for new spatial operations.

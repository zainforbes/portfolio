# Generative AI & Routing Demo

A lightweight Python demo and rapid-prototype CLIP notebook that shows how to securely load your ORS (OpenRouteService) and OpenAI API keys from a `.env` file and make calls to the OpenAI API. Ideal for portfolio inclusion when showcasing best practices around secrets management, API integration, and prototype testing of CLIP models for location intelligence.

---

## 📦 Features

- Loads environment variables securely using [python-dotenv](https://github.com/theskumar/python-dotenv)
- Demonstrates setting up the OpenAI Python client
- Illustrates where to wire in an ORS (OpenRouteService) client
- Includes a Jupyter notebook for rapid prototyping OpenAI CLIP models applied to location intelligence, leveraging GPT as an LLM for deriving insights
- Provides guidance on security best practices

---

## 🔧 Requirements

- Python 3.7+
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [openai](https://pypi.org/project/openai/)
- [torchvision](https://pypi.org/project/torchvision/) and [torch](https://pypi.org/project/torch/) for CLIP notebook
- [jupyterlab](https://pypi.org/project/jupyterlab/) to run the notebook

---

## 🚀 Installation

1. **Clone this repo**  
   ```bash
   git clone https://github.com/yourusername/your-portfolio-repo.git
   cd your-portfolio-repo
   ```

2. **Create and activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS / Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install python-dotenv openai torch torchvision jupyterlab
   ```

---

## ⚙️ Configuration

1. **Add your keys to `.env`**  
   Create a file named `.env` in the project root with the following entries (no quotes):  
   ```bash
   ORS_API_KEY=your-openrouteservice-key-here
   OPENAI_API_KEY=your-openai-key-here
   ```

2. **Ensure `.env` is git-ignored**  
   Make sure your `.gitignore` contains:
   ```
   # Secrets
   .env
   ```

---

## 📄 Usage

### Python API Demo

```python
from dotenv import load_dotenv
import os
import openai

# 1. Load variables from .env into process environment
load_dotenv()

# 2. Retrieve keys
ors_api_key  = os.getenv("ORS_API_KEY")
open_api_key = os.getenv("OPENAI_API_KEY")

# 3. Configure the OpenAI client
openai.api_key = open_api_key

# (Optional) Configure ORS client, e.g.:
# import openrouteservice
# client = openrouteservice.Client(key=ors_api_key)

# 4. Make an OpenAI API call
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
print(response.choices[0].message.content)
```

### CLIP Location Intelligence Notebook

Launch the notebook to explore:

```bash
jupyter lab RapidPrototype_CLIP_Location_Intel.ipynb
```

In the notebook:
1. Load and preprocess images and geospatial data.
2. Use OpenAI CLIP to embed and compare image-text pairs.
3. Query GPT for insights on location-based patterns and features.
4. Visualize results directly in the notebook.

---

## 🔒 Security Considerations

- **Never commit** your `.env` or any file containing secrets to version control.
- Use a secure vault (e.g., AWS Secrets Manager, HashiCorp Vault) for production deployments.
- Rotate your API keys regularly.
- Follow the principle of least privilege—generate separate keys with minimal scopes when available.
- Review your `.gitignore` to ensure no other credential files are accidentally tracked.

---

## 🏷️ License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙋‍♂️ About Me

Feel free to explore the rest of my portfolio for more projects on generative AI, geospatial analysis, and full-stack development.  
– **Zain Forbes**
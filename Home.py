import streamlit as st

st.set_page_config(page_title="Ollama Llama3 Project", page_icon="🦙", layout="wide")

st.title("Welcome to the Ollama Llama3 Project")

st.write("""
## About This Project

This Streamlit app showcases the integration of Ollama running the Llama3 8B model. 
Our project aims to demonstrate the capabilities of large language models in a user-friendly interface.

### What is Ollama?

Ollama is an open-source project that allows you to run large language models locally. 
It provides a simple way to run, manage, and customize various AI models on your own hardware.

### Llama3 8B Model

We're using the Llama3 8B model, which is a powerful language model developed by Meta AI. 
It's capable of understanding and generating human-like text across a wide range of topics and tasks.

### Key Features

- **Local Processing**: All computations are done on your local machine, ensuring privacy and control over your data.
- **Customizable**: Ollama allows for easy customization and fine-tuning of the model to suit specific needs.
- **Fast Inference**: The 8B parameter version of Llama3 offers a good balance between performance and resource requirements.

## Getting Started

To use this project, you need to have Docker installed and Ollama running in a Docker container. Follow these steps:

1. Pull the Ollama Docker image:
   ```
   docker pull ollama/ollama
   ```

2. Start Ollama with the Llama3 model:
   ```
   docker exec -it ollama ollama run llama3
   ```

Once Ollama is running, you can interact with the model through various interfaces we've built in this Streamlit app.

**Note**: This project was developed and tested on a system with an RTX4060 GPU.

## Explore Further

Use the navigation menu to explore different features and demonstrations of the Llama3 model's capabilities.
""")

st.sidebar.header("Navigation")
st.sidebar.write("(Add links to github page later)")

st.sidebar.header("Resources")
st.sidebar.markdown("""
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Llama3 Information](https://ai.meta.com/llama/)
- [Streamlit Documentation](https://docs.streamlit.io/)
""")

st.sidebar.header("About the Developers")
st.sidebar.write("Created with ❤️ by Joseph Ho")
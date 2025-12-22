# Indian Sign Language (ISL) Translation System

A real-time Indian Sign Language translation system that combines BiLSTM networks with Large Language Models to bridge the communication gap for the deaf community.

## Overview
With approximately 63 million people in India suffering from significant hearing loss but only 339 certified ISL interpreters nationwide, there's an urgent need for automated translation solutions. This project addresses that gap using deep learning and modern language models.

### Key Features

- **Real-time Sign Detection**: Live webcam-based sign language recognition
- **Continuous Translation**: Converts sequences of signs into natural English sentences
- **Dual Modes:**
  - **Word Detection**: Practice individual signs with confidence scores
  - **Sentence Formation**: Build and translate complete sentences
- **Resource Efficient**: Trained on only 30 signs, achieving 96.35% accuracy
- **LLM Integration**: Uses Llama 3.1 for natural language generation

<div align="center">
  <img src="https://github.com/user-attachments/assets/04a0b442-0ed2-40a8-9735-af5e8a77f35b" alt="Flowchart">
  <br>
  <em>Figure 1: System flowchart</em>
</div>

<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/1b0047ff-d898-4d19-98a9-1ae87458d746" alt="Model Architecture">
  <br>
  <em>Figure 2: Model architecture.</em>
</div>


## Performance

| Metric | Score |
|--------|-------|
| Isolated Sign Accuracy | **96.35%** |
| F1-Score | **0.963** |
| BLEU-4 (Translation) | **54.1%** |
| BLEURT Score | **56.08%** |

## Dataset

This project uses a subset of the [**INCLUDE dataset**](https://zenodo.org/records/4010759) - a large-scale Indian Sign Language dataset containing over 4,000 videos across 263 unique ISL signs.

**Our Subset:**
- 30 commonly used signs across 7 categories
- 269 original videos (8-12 per sign)
- Augmented to ~100 videos per sign
- Categories: Greetings, Pronouns, Adjectives, Occupations, Objects, Places, Time

**Supported Signs:**
- Greetings: Hello, Good Morning, Goodnight, Thank you
- Pronouns: I, You, He, She, It
- Occupations: Doctor, Teacher, Lawyer
- Adjectives: Nice, Mean, Expensive, Poor, Thick, Thin, Pleased, Alright, Patient
- Descriptors: Male, Female
- Questions: How are you


## Setup

### Prerequisites
- Python 3.8+
- Webcam
- GROQ API key ([Get one here](https://console.groq.com/keys))

### Installation
```bash
# Clone the repository
git clone https://github.com/reubenjrouse/SignTalk.git
cd SignTalk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_api_key_here" > .env
```

### Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Demo
Coming Soon

## Project Structure
```
SignTalk/
├── app.py                      # Main Streamlit application
├── models/
│   ├── model.py               # BiLSTM model architecture
│   └── best_model_24.pth      # Trained model weights
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── README.md
```

## Authors

- [@reubenjrouse](https://github.com/reubenjrouse)
- [@leharrathore25](https://github.com/leharrathore25)

## Acknowledgements

- **INCLUDE Dataset**: Sridhar et al. for providing the Indian Sign Language dataset
- **Mukesh Patel School of Technology and Management, NMIMS University**


<div align="center">
Made with ❤️ for the Indian deaf community
</div>

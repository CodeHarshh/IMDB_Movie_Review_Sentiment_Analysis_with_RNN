
# IMDB Movie Review Sentiment Analysis with RNN

This repository contains a project for sentiment analysis of IMDB movie reviews using Recurrent Neural Networks (RNN). The primary aim of this project is to classify movie reviews as positive or negative based on their text content.

## Features

- **Text Preprocessing**: Includes tokenization and padding to prepare the data for model training and prediction.
- **Model Architecture**: Uses a Simple RNN model for sentiment classification.
- **User Interaction**: Streamlit app interface for real-time sentiment analysis.
- **Deployment**: Local deployment for testing or extension to cloud platforms.

---

## Project Structure

```plaintext
IMDB_Movie_Review_Sentiment_Analysis_with_RNN/
│
├── notebooks/                # Jupyter Notebooks for experiments
│   ├── data_processing.ipynb # Data preprocessing and exploration
│   ├── model_training.ipynb  # Model training and evaluation
│   └── model_analysis.ipynb  # Performance metrics and visualizations
│
├── app/                      # Streamlit application files
│   ├── app.py                # Main application script
│   └── requirements.txt      # Required dependencies
│
├── models/                   # Saved models
│   └── simple_rnn_model.h5   # Pre-trained RNN model
│
├── data/                     # Data files
│   ├── train/                # Training data
│   └── test/                 # Testing data
│
├── README.md                 # Project documentation
└── LICENSE                   # License information
```

---

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository**  
   Clone the repository to your local system:
   ```sh
   git clone https://github.com/CodeHarshh/IMDB_Movie_Review_Sentiment_Analysis_with_RNN.git
   cd IMDB_Movie_Review_Sentiment_Analysis_with_RNN
   ```

2. **Create and Activate a Virtual Environment**  
   Set up a virtual environment for dependency management:
   ```sh
   python -m venv venv
   source venv/bin/activate    # On macOS/Linux
   venv\Scripts\activate       # On Windows
   ```

3. **Install Dependencies**  
   Install the required Python libraries:
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Application**  
   Launch the Streamlit app:
   ```sh
   streamlit run app/app.py
   ```

5. **Test the Application**  
   Open your browser at the URL provided by Streamlit (e.g., `http://localhost:8501`) and input movie reviews to classify their sentiment.

---

## Usage

- **Input**: Enter a movie review into the Streamlit application.
- **Output**: The model predicts whether the sentiment is "Positive" or "Negative" along with a confidence score.

---

## Model Details

- **Model Type**: Simple RNN with embedding and dense layers.
- **Training Data**: IMDB dataset of 25,000 movie reviews.
- **Evaluation**: Achieved an accuracy of ~85% on the test dataset.

---

## Requirements

The required libraries are listed in `requirements.txt`. Key dependencies include:
- Python 3.7+
- TensorFlow
- Streamlit
- NumPy
- h5py

---

## Future Enhancements

- Implementing more advanced models like LSTM or GRU for better performance.
- Integrating a larger dataset for improved generalization.
- Deploying the application to cloud services like AWS or Heroku.

---

## Contributing

Contributions are welcome! If you would like to contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset provided by [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/).
- TensorFlow and Streamlit communities for their resources and support.


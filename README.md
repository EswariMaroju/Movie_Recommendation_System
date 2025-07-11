# Movie Recommendation System using Hybrid Deep Learning Models

-----

## Project Description

This project develops an advanced **Movie Recommendation System** that leverages a **hybrid deep learning approach** to deliver highly personalized movie suggestions. It strategically combines the strengths of **Neural Collaborative Filtering (NCF)**, **Autoencoders**, and **content-based filtering** (utilizing movie genres and tags) to significantly enhance recommendation accuracy.

Built with **Python** and **TensorFlow**, the system efficiently processes vast datasets of movie ratings, user tags, and comprehensive movie metadata. This data is then used to train sophisticated models capable of accurately predicting individual user preferences, ultimately providing more relevant and engaging movie recommendations.

-----

## Installation

Follow these detailed steps to set up the project environment and install all necessary dependencies:

### 1\. Install Python

Ensure you have **Python 3.7 or higher** installed on your system. You can download the latest version from the official Python website:
[https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2\. Create a Virtual Environment (Recommended)

It is highly recommended to create a virtual environment to isolate project dependencies and avoid conflicts with other Python projects:

```bash
python -m venv venv
```

Activate the virtual environment:

  * **On Windows:**
    ```bash
    venv\Scripts\activate
    ```
  * **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### 3\. Install Required Libraries

With your virtual environment activated, install the necessary Python libraries using `pip`:

```bash
pip install tensorflow pandas scikit-learn numpy
```

### 4\. Prepare Data Files

Ensure the following essential CSV files are present in your project's root directory:

  * `movies.csv`
  * `ratings.csv`
  * `tags.csv`

-----

## Usage

To run the movie recommendation system and generate recommendations, follow these steps:

1.  **Navigate to Project Directory:** Open your terminal or command prompt and change the directory to where your project files are located.

2.  **Activate Virtual Environment:** Ensure your virtual environment is activated (if you created one, as recommended in the installation steps).

3.  **Run the Main Script:** Execute the main script using Python:

    ```bash
    python recommend.py
    ```

The script will automatically perform the following actions:

  * Load and preprocess the movie data.
  * Train multiple deep learning models, including Neural Collaborative Filtering, Autoencoder, and the combined Hybrid model.
  * Output movie recommendations.
  * Display various evaluation metrics to assess model performance.

You will observe printed outputs detailing the training progress, evaluation results, and the top recommended movies for different users.

-----

## Features

This recommendation system boasts the following key features:

  * **Hybrid Recommendation Approach:** Integrates collaborative filtering (Neural Collaborative Filtering), autoencoders, and content-based filtering (using movie genres and tags) to achieve superior recommendation accuracy.
  * **Robust Data Preprocessing:** Efficiently merges and processes raw movie ratings, tags, and metadata into a usable format for model training.
  * **Advanced Model Training:** Trains and fine-tunes multiple deep learning models, including specialized Neural Collaborative Filtering architectures and Autoencoders.
  * **Intelligent Hybrid Model:** Combines the outputs and strengths from different individual models to generate more precise and diverse final recommendations.
  * **Comprehensive Evaluation Metrics:** Provides industry-standard evaluation metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) to quantitatively assess model performance.
  * **Top-N Recommendations:** Generates a list of top movie recommendations for each user based on their predicted preferences.

-----

## Technologies Used

The project is built upon a robust stack of technologies:

  * **Python 3.7+:** The primary programming language.
  * **TensorFlow & Keras:** Essential frameworks for building and training deep learning models.
  * **pandas:** Utilized for powerful data manipulation and analysis.
  * **scikit-learn:** Employed for various data splitting, preprocessing, and machine learning utilities.
  * **NumPy:** Fundamental library for numerical operations and array processing.

-----

## License

This project is licensed under the **MIT License**. For complete details, please refer to the [LICENSE](https://www.google.com/search?q=LICENSE) file located in the project's root directory.

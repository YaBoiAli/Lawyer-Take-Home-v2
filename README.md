# OCR and GraphRAG Project

This project combines Optical Character Recognition (OCR) with a Graph-based Retrieval Augmented Generation (RAG) system.

## How to Run the Project
### Without Docker

To run this project without Docker, follow these steps:

1.  **Create and Activate Virtual Environment**:
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    With your virtual environment activated, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**:
    Create a `.env` file in the root directory of the project. This file should contain the following environment variables:
    *   `NEON_DB_URL`: Your Neon database connection URL.
    *   `OPENAI_API_KEY`: Your OpenAI API key.
    
    Example `.env` content:
    ```
    GROQ_API_KEY=
    NEON_CONNECTION_STRING=""
    NEO4J_URI=""
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=
    ```

4.  **Run the Application**:
    Start the Flask application:
    ```bash
    python app.py
    ```
    The application should now be running, typically accessible at `http://127.0.0.1:5000`.

### With Docker

To run this project using Docker and Docker Compose, follow these steps:

1.  **Install Docker**:
    Ensure you have Docker and Docker Compose installed on your system. You can download them from the official Docker website.

2.  **Set up Environment Variables**:
    Create a `.env` file in the root directory of the project and add necessary environment variables as described in the "Without Docker" section above. These variables will be picked up by Docker Compose.

3.  **Build and Run with Docker Compose**:
    Navigate to the root directory of the project in your terminal and run:
    ```bash
    docker-compose up --build
    ```
    This command will build the Docker image (if it hasn't been built yet or if changes were made to the `Dockerfile`) and start the services defined in `docker-compose.yml`. The application should then be accessible, typically at `http://localhost:5000`.

4.  **Stop the Application**:
    To stop the running Docker containers, press `Ctrl+C` in the terminal where `docker-compose up` is running. To remove the containers, networks, and volumes created by `up`, run:
    ```bash
    docker-compose down
    docker-compose down -v      Deletes everything saved locally via volumes
    ```

## Project Modules

The project is structured into several modules, each handling specific functionalities:

*   `app.py`: The main Flask application file, handling routes and overall application logic.
*   `modules/ocr_module.py`: Contains logic for Optical Character Recognition.
*   `modules/rag_module.py`: Implements the Retrieval Augmented Generation logic.
*   `modules/neon_db_module.py`: Manages interactions with the Neon database.
*   `static/`: Contains static files like `script.js` (JavaScript) and `styles.css` (CSS).
*   `templates/`: Stores HTML templates, such as `index.html`.

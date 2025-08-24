%%markdown
### Instructions to Run Jupyter Notebook in a Docker Container

To run the Jupyter notebook in a containerized environment using the generated files, follow these steps:

1.  **Build the Docker Image:**
    Open your terminal or command prompt in the directory where the `Dockerfile`, `docker-compose.yml`, and `requirements.txt` files are located.
    Run the following command to build the Docker image. This command will read the `Dockerfile` and create an image named `jupyter-notebook`.

    ```bash
    docker build -t jupyter-notebook .
    ```

2.  **Run the Docker Container:**
    After the image is built, use `docker-compose` to run the container in detached mode (in the background). This will create and start a service named `jupyter` as defined in the `docker-compose.yml` file, mapping port 8888 on your host to port 8888 in the container.

    ```bash
    docker-compose up -d
    ```

3.  **Access the Jupyter Notebook:**
    The Jupyter notebook server is now running inside the container. To access it, open a web browser and go to:

    ```
    http://localhost:8888
    ```

    You will be prompted to enter a token. To get the token, view the logs of the running container using the following command:

    ```bash
    docker logs jupyter_notebook_container
    ```

    Look for a line in the output that contains the URL with the token, similar to:
    `http://127.0.0.1:8888/?token=YOUR_TOKEN_HERE`
    Copy the `YOUR_TOKEN_HERE` part and paste it into the token field in your browser.

4.  **Stop the Docker Container:**
    When you are finished, you can stop the running container using `docker-compose down` in your terminal in the same directory:

    ```bash
    docker-compose down
    ```
# zomato_recommendation

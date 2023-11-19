# ID2223Lab1

To run the iris (and wine) notebooks I used the docker all-spark jupyter nootebook docker image for python 3.10 (I installed hopsworks from within the notebook). I ran the "daily" feature and inference pipelines 4 times a day (in order to have a slightly faster dataflow and see if it worked correctly quicker) using github actions instead of modal. The two apps were run on hugggingface gradio spaces, I had too change the requirements
to include httpx==0.24.1 but appart from that I simply had to copy from the example code.

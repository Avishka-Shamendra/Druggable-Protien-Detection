## Prerequisites
Python 3.9 or later

## Setup a Vritual Environment

1. Open a terminal or command prompt and run the following command to create a virtual environment named "venv":
    ```
    python -m venv venv
    ```
2. Activate the environment

      Windows:
        ```
        venv\Scripts\activate
        ```
      Ubuntu/MacOS:
        ```
        source venv/bin/activate
        ```
4. Run
    ``` 
    pip install -r requirements.txt
    ```

## Run

To train and test the dataset

```
python script.py ./dataset/TR_pos_SPIDER.txt ./dataset/TR_neg_SPIDER.txt ./dataset/TS_pos_SPIDER.txt ./dataset/TS_neg_SPIDER.txt
```

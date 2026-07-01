# Execution

## Local 

To run the project locally, you can use the following command:

```bash
uv venv 
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
uv pip install -r requirements.txt
```

Run the script:

```bash
python main.py
```

## Docker

```bash
docker build -t tennis-match-predictor .
docker run --rm -it tennis-match-predictor
```
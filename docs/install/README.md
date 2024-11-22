# Install the project

## Windows

Run the following code snippet tocCreate the virtualenv (you can also use conda) and install the dependencies of `requirements.txt`

```sh
python -m venv env
env/Scripts/activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
mkdir data
```

## Linux
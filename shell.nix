{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "aspect-sentiment-env";

  packages = [
    (pkgs.writeShellScriptBin "nvidia-smi" "")
  ];

  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.virtualenv
    # pkgs.rustup
    # pkgs.darwin.libiconv
    # pkgs.oniguruma
  ];

  shellHook = ''
    # Create and activate a virtual environment
    if [ ! -d "rfc-venv" ]; then
      virtualenv rfc-venv
    fi
    source ./rfc-venv/bin/activate

    pip install -r requirements.txt

    # pip install "numpy<2"
    # pip install -U pyabsa
    # pip install -U torch transformers

    # pip install praw

    # python -m spacy download en_core_web_sm
  '';
}

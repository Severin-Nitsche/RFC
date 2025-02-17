{
  description = "RFC python environments";

  inputs = {
    nixpkgs-3_10_8.url = "github:nixos/nixpkgs/79b3d4bcae8c7007c9fd51c279a8a67acfa73a2a";
    nixpkgs-3_12_8.url = "github:nixos/nixpkgs/21808d22b1cda1898b71cf1a1beb524a97add2c4";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs-3_10_8, nixpkgs-3_12_8, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let 
        pkgs-3_10_8 = import nixpkgs-3_10_8 { inherit system; };
        pkgs-3_12_8 = import nixpkgs-3_12_8 { inherit system; };
      in {
        devShells.default = pkgs-3_10_8.mkShell {
          name = "Python 3.10.8 environment";
          packages = [
            (pkgs-3_10_8.writeShellScriptBin "nvidia-smi" "")
          ];
          buildInputs = [ 
            pkgs-3_10_8.python3
            pkgs-3_10_8.python3Packages.pip
            pkgs-3_10_8.python3Packages.virtualenv
          ];
        };
        devShells.label-studio = pkgs-3_12_8.mkShell {
          name = "Python 3.12.8 environment (label-studio)";
          buildInputs = [ 
            pkgs-3_12_8.python3
            pkgs-3_12_8.python3Packages.pip
            pkgs-3_12_8.python3Packages.virtualenv
          ];
        };
      }
    );
}

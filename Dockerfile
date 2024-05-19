# Használjunk egy alap Ubuntu image-et
FROM ubuntu:20.04

# Frissítsük a csomaglistát és telepítsük a szükséges csomagokat
RUN apt-get update && apt-get install -y \
    cmake \
    make \
    pandoc \
    texlive-full \
    && rm -rf /var/lib/apt/lists/*

# Állítsuk be a munka könyvtárat
WORKDIR /workspace

# Másoljuk át a repository tartalmát a munka könyvtárba
COPY . /workspace

# Definiáljuk az alapértelmezett parancsot (ezt a GitHub Actions workflow-ban felülírjuk)
CMD ["bash"]

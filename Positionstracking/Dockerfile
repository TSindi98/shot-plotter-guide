FROM node:16-alpine

# Labels für das Image
LABEL version="1.3"
LABEL description="Shot-Plotter - Football/Soccer Pass Tagging and Analysis Tool"
LABEL maintainer="sindi98"
LABEL repository="sindi98/shot-plotter-shot-plotter"

WORKDIR /app

# Installiere http-server global
RUN npm install -g http-server

# Kopiere alle notwendigen Dateien
COPY index.html .
COPY html/ ./html/
COPY js/ ./js/
COPY resources/ ./resources/
COPY preprocessing/ ./preprocessing/
COPY supported-sports.json .
COPY setup.js .
COPY _redirects .
COPY *.md ./

# Setze Berechtigungen
RUN chmod -R 755 /app

# Exponiere Port 8080
EXPOSE 8080

# Healthcheck hinzufügen
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/ || exit 1

# Starte den http-server
CMD ["http-server", ".", "-p", "8080", "--cors", "-c-1"]

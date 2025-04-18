FROM node:16-alpine

WORKDIR /app

# Kopiere alle Dateien aus dem aktuellen Verzeichnis
COPY . .

# Installiere http-server global
RUN npm install -g http-server

# Exponiere Port 8080
EXPOSE 8080

# Starte den http-server
CMD ["http-server", "-p", "8080", "--cors"]

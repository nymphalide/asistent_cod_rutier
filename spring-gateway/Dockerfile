# Standard Java 21 Runtime for the Spring Boot service [cite: 6]
FROM eclipse-temurin:21-jre-alpine

# Install netcat to handle dummy port listening
RUN apk add --no-cache netcat-openbsd

WORKDIR /app

# Placeholder for the eventual JAR file [cite: 106]
# Once you build the app, you'll change this to: COPY target/*.jar app.jar

EXPOSE 8080

# Dummy loop: Keeps the container running and listening for pings from the frontend [cite: 88]
CMD while true; do echo -e "HTTP/1.1 200 OK\n\n Spring Gateway Placeholder" | nc -l -p 8080; done
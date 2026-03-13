# High-performance Nginx to serve the React frontend 
FROM nginx:alpine

# Create a basic placeholder page until the React build is ready [cite: 5]
RUN echo "<html><body><h1>Traffic Guru Frontend Placeholder</h1><p>Waiting for React build...</p></body></html>" > /usr/share/nginx/html/index.html

# Map to the port used by the browser [cite: 78]
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
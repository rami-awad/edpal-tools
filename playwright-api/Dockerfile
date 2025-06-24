# Use the official Node.js image
FROM mcr.microsoft.com/playwright:v1.52.0-jammy

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install
RUN npm install --include=dev

# Copy rest of the app
COPY . .

# Expose the port your app runs on
EXPOSE 3000

# Start the server
CMD ["npm", "start"]
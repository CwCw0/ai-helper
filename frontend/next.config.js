const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  output: "standalone",
  webpack: (config) => {
    // Make '@' point to the frontend root folder
    config.resolve.alias["@"] = path.resolve(__dirname);
    return config;
  },
};
module.exports = nextConfig;

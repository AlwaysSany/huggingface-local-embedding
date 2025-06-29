# Security Policy

## Supported Versions

We release security updates for the latest major version. Please ensure you are using the latest version of this project for the best security.

| Version | Supported          |
| ------- | ----------------- |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing [sany2k8@gmail.com](mailto:sany2k8@gmail.com) or by creating a private GitHub issue. We will respond as quickly as possible to address the issue.

**Please do not disclose security issues publicly until they have been resolved.**

## Security Best Practices

- **Keep dependencies up to date:** Regularly update your Python packages and Docker images to receive the latest security patches.
- **Environment Variables:** Never commit secrets (API keys, tokens, etc.) to version control. Use `.env` files or environment variables for sensitive data.
- **ngrok Token:** Treat your ngrok token as a secret. Do not share it or expose it in public repositories.
- **Hugging Face Tokens:** If using private Hugging Face models, keep your `HF_TOKEN` secret.
- **API Exposure:** If deploying publicly, consider adding authentication (e.g., API keys, OAuth) to protect your endpoints.
- **CORS:** Configure CORS settings in FastAPI if you expect cross-origin requests.
- **HTTPS:** Use HTTPS in production. If using ngrok, it provides HTTPS by default.
- **File Uploads:** The server allows file uploads. Validate and sanitize all uploaded files to prevent malicious content.
- **Resource Limits:** For Docker deployments, consider setting resource limits to prevent abuse.

## Responsible Disclosure

We appreciate responsible disclosure of security vulnerabilities. Please give us a reasonable time to address the issue before any public disclosure.

Thank you for helping keep this project and its users safe! 
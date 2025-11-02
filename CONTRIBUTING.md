# 🤝 Contributing to FindMyJoint

Thanks for your interest in contributing to **FindMyJoint**!  
This project welcomes all kinds of contributions — from bug fixes to documentation improvements.

---

## 🧩 Getting Started

1. **Fork** the repository  
2. **Clone** your fork:
   ```bash
   git clone https://github.com/Rudra-G-23/Find-My-Joint.git
   cd Find-My-Joint
   ```

3. Create a branch for your feature or fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```

4. Install dependencies:
   
    ```bash
    pip install -e .[dev]
    ```

## 🧪 Running Tests

Please make sure your changes pass all tests before submitting a PR.

```bash
pytest
```

## 💡 Pull Request Guidelines

- Keep PRs small and focused
- Use clear, descriptive commit messages
- Include screenshots or outputs when relevant
- Update docs and type hints if you modify core functions

## Code Style
Try to follow the PEP8 and use

```bash
black .
flake 8
```
# Contributing to Noventis

First off, thank you so much for your interest in contributing to Noventis! We're thrilled you're here. Every contribution, no matter how small, is greatly appreciated and helps us make this tool better for everyone.

This guide provides a set of guidelines and steps to make your contribution process as easy and effective as possible.

---

## Code of Conduct

To maintain a friendly and inclusive community, this project and all its participants are governed by the Noventis Code of Conduct. Please adhere to this code in all your interactions with the project.

---

## How Can I Contribute?

There are many ways to contribute, and not all of them involve writing code.

### 🐛 Reporting Bugs

If you find something that isn't working as expected, please open a new issue on our [GitHub Issues page](https://github.com/Noventis-Laplace-Project/Noventis-Data//issues). 

**When reporting bugs, please include:**
- Steps to reproduce the bug
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, Noventis version)
- Error messages or stack traces (if applicable)

### 💡 Suggesting New Features

Have a brilliant idea for a new functionality? We'd love to hear it! Open a new issue and describe your idea in detail.

**When suggesting features, please include:**
- What problem does it solve?
- How should it work?
- Any examples or use cases

### 📝 Improving Documentation

Found a typo or a confusing sentence in our documentation? Fixes to documentation are just as important as fixes to code.

### 🔧 Submitting Pull Requests

If you want to add a feature, fix a bug, or improve documentation, this is the best way to do it. Follow the workflow below!

---

## Your Contribution Workflow

Ready to start contributing? Follow these steps to set up your development environment and submit your first change.

### Step 1: Fork the Repository

Click the "Fork" button at the top right corner of the [Noventis GitHub page](https://github.com/Noventis-Laplace-Project/Noventis-Data/) to create a copy of the repository in your own GitHub account.

### Step 2: Clone Your Fork

Now, clone your fork to your local machine.

```bash
git clone https://github.com/YOUR-USERNAME/noventis.git
cd noventis
```

### Step 3: Create a Virtual Environment

Always work inside a virtual environment to keep dependencies isolated.

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or .\venv\Scripts\activate for Windows
```

### Step 4: Install Dependencies

Install the library in "editable" mode (`-e`). This means changes you make to the source code will take effect immediately without needing to reinstall.

```bash
pip install -e .
```

### Step 5: Create a New Branch

Create a new branch to work on your changes. Use a descriptive name.

**Branch naming conventions:**
- For new features: `feat/new-feature-name`
- For bug fixes: `fix/short-bug-description`
- For documentation: `docs/description`

```bash
git checkout -b feat/add-svm-model
```

### Step 6: Write Your Code!

Time to work your magic! Make your changes, add your features, or fix the bug. Be sure to follow our **Coding Standards** (see below).

### Step 7: Commit Your Changes

Use a clear and descriptive commit message. We recommend the [Conventional Commits](https://www.conventionalcommits.org/) format.

**Commit types:**
- `feat:` for a new feature
- `fix:` for a bug fix
- `docs:` for documentation changes
- `test:` for adding or updating tests
- `refactor:` for code refactoring
- `style:` for code style changes

```bash
git add .
git commit -m "feat: add support for SVM model in ManualPredictor"
```

### Step 8: Push to Your Fork

Upload your changes to your forked repository on GitHub.

```bash
git push origin feat/add-svm-model
```

### Step 9: Submit a Pull Request (PR)

1. Open your forked repository on GitHub
2. You will see a button to create a **Pull Request** - click it
3. Provide a clear title and description for your PR
4. Reference any related issues (e.g., "Fixes #123")
5. Submit the pull request

---

## Coding Standards

To maintain code quality and consistency, please follow these standards:

### Code Style

We use **Black** for automated code formatting. Please run `black .` before committing to automatically format your code.

```bash
black .
```

### Docstrings

All new public functions, classes, and methods must have a clear docstring explaining their purpose, parameters, and what they return.

**Example:**
```python
def predict(self, X):
    """
    Make predictions on input data.
    
    Args:
        X (array-like): Input features of shape (n_samples, n_features)
        
    Returns:
        array: Predictions of shape (n_samples,)
        
    Raises:
        ValueError: If model is not fitted
    """
    pass
```

### Testing

If you add new functionality, it's highly encouraged to add corresponding unit tests. Contributions that include tests are prioritized.

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=noventis
```

---

## Pull Request Review Process

Once you submit a PR, one of the project maintainers will review it. The process is as follows:

### 1. Initial Review
We will check if your PR is clear and aligns with the project's goals.

### 2. Feedback
We may provide comments or request some changes to improve the code quality.

### 3. Approval & Merge
Once all feedback is addressed and all automated checks (CI) have passed, your PR will be merged into the main branch.

**Congratulations!** 🎉 Your contribution is now a part of Noventis.

---

## Need Help?

If you have questions or need assistance:
- Open an issue on GitHub
- Check existing documentation and issues
- Reach out to the maintainers

---

Once again, thank you for being a part of the Noventis community! Your contributions help make this project better for everyone. 🙏

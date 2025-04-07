
# **EdPal Tools Setup Guide**

This guide walks you through setting up the environment and installing dependencies so you can get started quickly.

## **1. Prerequisites**
Ensure you have the following installed before proceeding:
- Python 3.10+
- PowerShell or Command Prompt (for running commands)
- `pip` (Python package manager)
- `virtualenv` (comes with Python 3.10+)

## **2. Setting Up the Virtual Environment**
To isolate dependencies, create a virtual environment:

```
python -m venv edpal-tools-env
```
Activate the environment:
```
edpal-tools-env\Scripts\activate
```
To deactivate it when you're done:
```
deactivate
```

## **3. Installing Dependencies**
Ensure all necessary libraries are installed:
```
pip install -r requirements.txt
```

## **4. Running the Application**
Once your environment is set up, you can execute your script:
```
python Contacts\script_name.py
```

## **5. Managing Dependencies**
If new packages are installed, update :
```
pip freeze > requirements.txt
```

## **6. Removing the Virtual Environment**
To delete the environment and start fresh:
```
Remove-Item -Recurse -Force edpal-tools-env
```

## **7.  Troubleshooting**
* If a package is missing, reinstall it:
  ```
  pip install <package-name>
  ```
 * If using VSCode, ensure the correct Python interpreter is selected:
   * Ctrl + Shift + P → Python: Select Interpreter → Choose ```edpal-tools-env```
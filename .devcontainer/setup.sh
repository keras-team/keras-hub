pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
echo "sh shell/lint.sh" > .git/hooks/pre-commit
chmod a+x .git/hooks/pre-commit

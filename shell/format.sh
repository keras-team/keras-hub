isort --sl .
black --line-length 80 .

for i in $(find keras_nlp tests -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo $i
    cat shell/copyright.txt $i >$i.new && mv $i.new $i
  fi
done

flake8 .

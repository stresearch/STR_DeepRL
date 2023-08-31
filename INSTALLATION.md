## Install dependencies

There are two options:

A. (Recommended) Install with conda:

	1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

	```

	This install will modify the `PATH` variable in your bashrc.
	You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

	2. Create a conda environment that will contain python 3.9 or later
	```
	conda create -n rl_class python=3.9
	```

	3. activate the environment (do this every time you open a new terminal and want to run code):
	```
	source activate rl_class
	```

	4. Install the requirements into this conda environment

	To avoid issue of building box2d-py we will need to manually install swig first
	```
	pip install swig
	```

	```
	pip install -r requirements_colab.txt
	```

	5. Allow your code to be able to see 'rl_class'
	```
	cd <path_to_repo>
	$ pip install -e .
	```

This conda environment requires activating it every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.


B. Install on system Python:
	```
	pip install -r requirements_colab.txt
	```



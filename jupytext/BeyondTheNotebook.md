## Beyond the notebook

There are many reasons that Jupyter notebooks are great for prototyping and interactive analysis, but there are also several important limitations:

1. Jupyter notebooks do not play well with version control. Any time the notebook is executed, the metadata get changed, and images are also stored as text within the notebook, so finding relevant code differences between versions can be challenging.

2. It's too easy for things to happen out of order, which can cause problems. [One analysis](https://blog.jetbrains.com/datalore/2020/12/17/we-downloaded-10-000-000-jupyter-notebooks-from-github-this-is-what-we-learned/) of almost ten million Jupyter notebooks from Github showed that 36% of notebooks had a non-linear execution order.

3. Jupyter does not easily support state-of-the-art coding tools such as linters, code formatting tools, and automated refactoring and coding assistance tools.

4. It is difficult to integrate code testing in the Jupyter workflow.

5. Jupyter notebooks are not optimal for use on interactive systems such as clusters.

## Using Jupytext

Fortunately there is a tool called [Jupytext](https://jupytext.readthedocs.io/en/latest/) that allows one to easily move between Jupyter notebooks and plain text code.

As an example, look at [Untitled.ipynb], which is a simple notebook containing a couple of code cells and a couple of Markdown cells. If we look at the content of the file, we will see that it contains much more than the content of the cells:

```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "662d1faa-f0ec-4ac5-9a71-35a830b8a113",
   "metadata": {},
   "source": [
    "## This is an example notebook"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c2210494-8f5d-4abb-996e-b3e0b948abf6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "607fca74-c11d-482a-bd00-23e79b586924",
   "metadata": {},
   "source": [
    "Include some rather unpythonic code: https://docs.quantifiedcode.com/python-anti-patterns/readability/using_an_unpythonic_loop.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ff821c34-0619-48a9-9875-b4ee78286f37",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 1\n",
      "1 2\n",
      "2 3\n"
     ]
    }
   ],
   "source": [
    "list = [1,2,3]\n",
    "\n",
    "# creating index variable\n",
    "for i in range(0,len(list)):\n",
    "    # using index to access list\n",
    "    le = list[i]\n",
    "    print(i,le)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fea42050-9862-4bb2-9171-32c3d4f7cdb8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

We can convert this to pure python using the command:

```
jupytext Untitled.ipynb --to py:percent
```

This now contains python code along with metadata from the Jupyter notebook stored as comments within the python file:

```
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## This is an example notebook

# %%
import os


# %% [markdown]
# Include some rather unpythonic code: https://docs.quantifiedcode.com/python-anti-patterns/readability/using_an_unpythonic_loop.html

# %%
list = [1,2,3]

# creating index variable
for i in range(0,len(list)):
    # using index to access list
    le = list[i]
    print(i,le)

# %%
```

## Code formatting

Open Untitled.py in Visual Studio Code. Click on the "Problems" tab, which should list a number of problems with the code formatting and structure. You may need to first enable the flake8 linter within VSCode; to do this, use Cmd-Shift-P to open the command palette, and then type "lint" and select "Python: Select linter" and choose _flake8_ from the list. You may also need to select "Python: Enable/disable linting" and set it to _enable_.

We can fix some of these using an autoformatting tool; we will use [blue](https://pypi.org/project/blue/), which is an adaptation of the popular autoformatter [black](https://black.readthedocs.io/en/stable/).

Run blue on the file using the command:

```
blue Untitled.py
```

It will change the file in place. If you reopen and then save the file in VSCode, you should see that the formatting problems are no longer present in the list.

You should also see a couple of problems raised by the _Sourcery_ extension (which you will need to install if you haven't already). This extension identifies potential refactorings that can be applied to the code to make it more Pythonic and robust.

Make the suggested fixes and save them to a new file called Untitled2.py.

### Converting back to a Jupyter notebook

We can convert the python file back to a Jupyter notebook using the command:

```
jupytext Untitled2.py --to ipynb
```

### Using Github copilot

#### Copilot labs features

- explain
- translate

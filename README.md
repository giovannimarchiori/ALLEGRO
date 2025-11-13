# k4-project-template


This repository can be a starting point and template for projects using the Key4hep software stack, in particular those writing Gaudi algorithms.
To use it, instead of forking it, go to Github and click `Use this template` at the top right.


## Dependencies

* ROOT

* PODIO

* Gaudi

* k4FWCore

## This repository as a template

If you belong to the Key4hep organization, then it is possible to use this
repository as a template for your new project, otherwise you can simply fork it
and follow the next steps.


### Compilation

Run, from the `k4-project-template` directory:

``` bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh
k4_local_repo
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -G Ninja
ninja install
```

Alternatively you can source the nightlies instead of the releases:

``` bash
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
```

Note that if you source the releases and use the current version of this
repository this is not guaranteed to work as there could be changes since this
repository was built for the release. What you can do in this case is to
checkout a previous tag, for example:

``` bash
git checkout v0.5.0
```

This is because the releases are only built with tagged versions of the
packages. With the nightlies this repository should always work; if it doesn't
please [open an
issue](https://github.com/key4hep/k4-project-template/issues/new/choose).

### Execute Examples

Make sure that `k4-project-template/install/lib` and
`k4-project-template/install/python` are in `LD_LIBRARY_PATH` and `PYTHONPATH`
respectively (`k4_local_repo` should take care of this). If they are not, they
can be added by running:
``` bash
export LD_LIBRARY_PATH=$PWD/install/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/install/python:$PYTHONPATH
```

and then run the examples like this:

``` bash
k4run k4ProjectTemplate/options/createHelloWorld.py
k4run k4ProjectTemplate/options/createExampleEventData.py
```

### Changing the name

Typically for a new project we would like to use a different name than
k4ProjectTemplate. The change of names can be done quickly running these
commands (change `k4CoolProject` to its name):

``` bash
mv k4ProjectTemplate k4CoolProject
git ls-files | xargs sed -i 's/k4ProjectTemplate/k4CoolProject/g'
```

After this, your project should be ready to run

### Continuous integration (CI) in forks
If this repository is forked instead of used as a template, CI will not work by
default, because GitHub disables workflows by default in forks of foreign
repositories. Workflows can be enabled by going to the Actions tab and clicking
the green button labeled `I understand my workflows, go ahead and enable them`.
If this is not done, no workflows will run, for example, after making a pull
request. In addition, scheduled workflows are also disabled by default and
require an additional step to be enabled manually from the Actions tab. The
workflows that are disabled will have `disabled` next to them (at the time of
writing, the only one is the `Key4hep build`).


## References:
These could perhaps be usefule for newcomers.
1. [lhcb-98-064 COMP](https://cds.cern.ch/record/691746/files/lhcb-98-064.pdf)
2. [Hello World in the Gaudi Framework](https://lhcb.github.io/DevelopKit/02a-gaudi-helloworld)

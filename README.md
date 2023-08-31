# flynet-looming
code for project: "Visual looming is a primitive for human emotion"

## Environment management notes

This repo manages Python package dependencies using conda. While conda tracks internal cross-package dependencies automatically when packages are installed using `conda install`, the environment contents must be manually exported to a YAML workflow file for recreation on other machines.

When exporting updated versions of the conda environment, export using the command `conda env export --from-history > environment.yml` to export only a lightweight environment file. The default behavior of `conda env export` is to export verrrry detailed information about every single package installed, including OS-specific dependency packages and package versions that might not be released for every OS. While this is the most exact documentation of package versions, it's basically impossible to use such a file to restore a comparable environment on a different operating system. As such, use the `--from-history` flag to record only packages that were explicitly installed by the user.

This isn't perfect, though--it appears you still need to manually add `- conda-forge` to the list of channels every time you re-export `environment.yml`, even when you manually install packages using the `conda install -c conda-forge` flag.

Please be aware that `--from-history` does _not_ record package versions unless you referenced a specific version when installing a package! Most of the time, we don't specify versions when installing packages to just install the most recent available, but it's kind of a pain that the installed version doesn't generally get logged when you export the environment this way. If you want to record the package version, you will need to either install with a version up front, or manually (I know, sorry) edit `environment.yml` to add `pkg=version.number`.

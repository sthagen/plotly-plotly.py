
# Release guide

## Release process - full release of `plotly` package

This is the release process for releasing `plotly.py` version `X.Y.Z`, including changelogs, Github release and forum announcement.

### Finalize changelog

Review the contents of `CHANGELOG.md`. We try to follow
the [keepachangelog](https://keepachangelog.com/en/1.0.0/) guidelines.
Make sure the changelog includes the version being published at the top, along
with the expected publication date.

Use the `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, and `Security`
labels for all changes to plotly.py.  If the version of plotly.js has
been updated, include this as the first `Updated` entry. Call out any
notable changes as sub-bullets (new trace types in particular), and provide
a link to the plotly.js CHANGELOG.

### Finalize versions

**Create a branch `git checkout -b release-X.Y.Z` *from the tip of `origin/master`*.**

Manually update the versions to `X.Y.Z` in the files specified below.

 - `CHANGELOG.md`
   + update the release date
 - `plotly/_widget_version.py`:
   + Update `__frontend_version__` to `^X.Y.Z` (Note the `^` prefix)
 - Commit your changes on the branch:
   + `git commit -a -m "version changes for vX.Y.Z"`

 ### Triggering (and Retriggering) the build

 - Commit and add this specific tag which `versioneer` will pick up, and push to Github so that CI will build the release artifacts. This is an atomic push so that CI will read the tag on the commit:
   + `git tag vX.Y.Z`
   + `git push --atomic origin release-X.Y.Z vX.Y.Z`
 - Create a Github pull request from `release-X.Y.Z` to `master` and wait for CI to be green
 - *If something goes wrong below*, you'll need to trigger the build process again after a fix. You'll need to commit your changes in the release branch, move the tag and atomically force push:
   + `git commit ....`
   + `git tag -f vX.Y.Z`
   + `git push --force --atomic origin release-X.Y.Z vX.Y.Z`

### Download and QA CI Artifacts

The `full_build` job in the `release_build` workflow in CircleCI produces a tarball of artifacts `output.tgz` which you should download and decompress, which will give you a directory called `output`. The filenames contained within will contain version numbers.

**Note: if any of the version numbers are not simply `X.Y.Z` but include some kind of git hash, then this is a dirty build and you'll need to clean up whatever is dirtying the tree and follow the instructions above to trigger the build again.** (That said, you can do QA on dirty builds, you just can't publish them.)

To locally install the PyPI dist, make sure you have an environment with JupyterLab installed (maybe one created with `conda create -n condatest python=3.10 jupyter anywidget pandas`):

- `tar xzf output.tgz`
- `pip uninstall plotly`
- `conda uninstall plotly` (just in case!)
- `pip install path/to/output/dist/plotly-X.Y.X-py3-none-any.whl`

To locally install the Conda dist (generally do this in a different, clean environment from the one above!):

- `conda uninstall plotly`
- `pip uninstall plotly` (just in case!)
- `conda install path/to/output/plotly-X.Y.Z.tar.bz2`

You'll want to check, in both Lab and Notebook, **in a brand new notebook in each** so that there is no caching of previous results, that `go.Figure()` and `go.FigureWidget()` work without error.

If something is broken, you'll need to fix it and trigger the build again (see above section).

### Publishing

Once you're satisfied that things render in Lab and Notebook in Widget and regular mode,
you can publish the artifacts. **You will need special credentials from Plotly leadership to do this.**.


Publishing to PyPI:
```bash
(plotly_dev) $ cd path/to/output/dist
(plotly_dev) $ twine upload plotly-X.Y.Z*
```

Publishing to `plotly` conda channel (make sure you have run `conda install anaconda-client` to get the `anaconda` command):

```
(plotly_dev) $ cd path/to/output
(plotly_dev) $ anaconda upload plotly-X.Y.Z.tar.bz2
```


### Merge the PR and make a Release

1. Merge the pull request you created above into `master`
2. Go to https://github.com/plotly/plotly.py/releases and "Draft a new release"
3. Enter the `vX.Y.Z` tag you created already above and make "Release title" the same string as the tag.
4. Copy the changelog section for this version as the "Describe this release"

### Update documentation site

1. Search for the previous version string in the docs and replace it with the new version string, including but not necessarily limited to the following files:
    - `doc/python/getting-started.md`
    - `doc/apidoc/conf.py`
    - `doc/requirements.txt`
    - `binder/requirements.txt`
2. `doc-prod` should already have been merged on a regular basis into `master`, but
start by doing it first if not. Then merge `master` into `doc-prod` to deploy the doc related
to features in the release.
3. in a clone of the [`graphing-library-docs` repo](https://github.com/plotly/graphing-library-docs):
    1. bump the version of Plotly.py in  `_data/pyversion.json`
    2. bump the version of Plotly.js with `cd _data && python get_plotschema.py <PLOTLY.JS VERSION>` fixing any errors that come up.
      - If Plotly.js contains any new traces or trace or layout attributes, you'll get a warning `“missing key in attributes: <attribute-name>`. To resolve, add the attribute to the relevant section in `/_data/orderings.json` in the position you want it to appear in the reference docs.
    3. rebuild the Algolia `schema` index with `ALGOLIA_API_KEY=<key> make update_ref_search`
    4. Rebuild the Algolia `python` index with `ALGOLIA_API_KEY=<key> make update_python_search`
    5. Commit and push the changes to `master` in that repo

### Notify Stakeholders

* Post an announcement to the Plotly Python forum, with links to the README installation instructions and to the CHANGELOG.
* Update the previous announcement to point to this one
* Update the Github Release entry and CHANGELOG entry to have the nice title and a link to the announcement
* Follow up on issues resolved in this release or forum posts with better answers as of this release

## Release process - Release *Candidate* of `plotly` package

(rough notes for a rough/ad hoc process!)

It's the same process as above except that the `X.Y.Z` version has a suffix and there are special instructions below for publishing an RC: note that the `npm` suffix is `-rc.1` and the PyPI suffix is `rc1`. We also don't update the docs with RC information and we inform a limited number of stakeholders.

PyPI RC (no special flags, just the `rc1` suffix):

```bash
(plotly_dev) $ twine upload dist/plotly-X.Y.Zrc1*
```

The `--tag next` part ensures that users won't install this version unless
they explicitly ask for the version or for the version with the `next` tag.

Conda RC:

```
$ anaconda upload --label test plotly-*.tar.bz2
```

The `--label test` part ensures that users won't install this version unless
they explicitly ask for the version or for the version with the `next` tag.

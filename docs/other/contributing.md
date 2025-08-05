# :material-arm-flex: **Contribution Guidelines**

We sincerely welcome contributions of any form to OmniGibson, as our aim is to make it a more robust and useful resource for the community. We have always held the belief that a collective effort from the community is essential to tailor BEHAVIOR/OmniGibson to meet diverse needs and unlock its full potential. 

## **Bug Reports & Feature Requests**

If you encounter any bugs or have feature requests that could enhance the platform, please feel free to open an issue on our GitHub repository. Before creating a new issue, we recommend checking the existing issues to see if your problem or request has already been reported or discussed. 

When reporting a bug, please kindly provide detailed information about the issue, including steps to reproduce it, any error messages, and relevant system details. For feature requests, we appreciate a clear description of the desired functionality and its potential benefits for the OmniGibson community.

You can also ask questions on our Discord channel about issues.

## **Branch Structure**

The OmniGibson repository uses the below branching structure:

* *main* is the branch that contains the latest released version of OmniGibson. No code should be directly pushed to *main* and no pull requests should be merged directly to *main*. It is updated at release time by OmniGibson core team members. External users are expected to be on this branch.
* *og-develop* is the development branch that contains the latest, stable development version. Internal users and developers are expected to be on, or branching from, this branch. Pull requests for new features should be made into this branch. **It is our expectation that og-develop is always stable, e.g. all tests need to be passing and all PRs need to be complete features to be merged.**

## **How to contribute**

We are always open to pull requests that address bugs, add new features, or improve the platform in any way. If you are considering submitting a pull request, we recommend opening an issue first to discuss the changes you would like to make. This will help us ensure that your proposed changes align with the goals of the project and that we can provide guidance on the best way to implement them.

**Before starting a pull request, understand our expectations. Your PR must:**

1. Contain clean code with properly written English comments
2. Contain all of the changes (no follow-up PRs), and **only** the changes (no huge PRs containing a bunch of things), necessary for **one** feature
3. Should leave og-develop in a fully stable state as you found it

You can follow the below items to develop a feature:

1. **Branch off of og-develop.** You can start by checking out og-develop and branching from it. If you are an OmniGibson team member, you can push your branches onto the OmniGibson repo directly. Otherwise, you can fork the repository.
2. **Implement your feature.** You can implement your feature, as discussed with the OmniGibson team on your feature request or otherwise. Some things to pay attention to:
  - **Examples:** If you are creating any new major features, create an example that a user can run to try out your feature. See the existing examples in the examples directory for inspiration, and follow the same structure that allows examples to be run headlessly as integration tests.
  - **Code style:** We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Please ensure that your code is formatted according to these guidelines. We have pre-commits that we recommend installing that fix style issues and sort imports. These are also applied automatically on pull requests.
  - **Inline documentation:** We request that all new APIs be documented via docstrings, and that functions be reasonably commented.
3. **Write user documentation**: If your changes affect the public API or introduce new features, please update the relevant documentation to reflect these changes. If you are creating new features, consider writing a tutorial.
4. **Testing**: Please include tests to ensure that the new functionality works as expected and that existing functionality remains unaffected. This will both confirm that your feature works, and it will protect your feature against regressions that can be caused by unrelated PRs by others. Unit tests are run on each pull request and failures will prevent PRs from being merged.
5. **Create PR**: After you are done with all of the above steps, create a pull request on the OmniGibson repo. **Make sure you are picking og-develop as the base branch.** A friendly bot will complain if you don't. In the pull request description, explain the feature and the need for changes, link to any discussions with developers, and assign the feature for review by one of the core developers.
6. **Go through review process**: Your reviewers may leave comments on things to be changed, or ask you questions. Even if you fix things or answer questions, do **NOT** mark things as resolved, let the reviewer do so in their next pass. After you are done responding, click the button to request another round of reviews. Repeat until there are no open conversations left.
7. **Merged!** Once the reviewer is satisfied, they will go ahead and merge your PR. The PR will be merged into og-develop for immediate developer use, and included in the next release for public use. Public releases happen every few months. Thanks a lot for your contribution, and congratulations on becoming a contributor to what we hope will be the world's leading robotics benchmark!

## **Continuous Integration**
The BEHAVIOR suite has continuous integration running via Github Actions in containers on our compute cluster. To keep our cluster safe, the CI will only be run on external work after one of our team members approves it.

* Tests and profiling are run directly on PRs and merges on the OmniGibson repo using our hosted runners
* Docker image builds are performed using GitHub-owned runners
* Docs builds are run on the behavior-website repo along with the rest of the website.
* When GitHub releases are created, a source distribution will be packed and shipped on PyPI by a hosted runner

For more information about the workflows and runners, please reach out on our Discord channel.

## **Release Process**
At the time of each release, we follow the below process:

1. Update the version of OmniGibson in the pyproject.toml and __init__.py files.
2. Add a release note on the README.md file
3. Push to `og-develop`
4. Wait for all tests to finish, confirm they are passing, confirm docs build on behavior-website
5. Push `og-develop` to `main`
6. Click on create release on GitHub, tag the version starting with the letter `v`
7. Create release notes. You can use the automated release notes but edit to include the important info.
8. Create the release.
9. Wait for docker and PyPI releases to finish, confirm success
10. Announce on Discord, user channels.
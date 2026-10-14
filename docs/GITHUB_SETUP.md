# GitHub publication checklist

## Suggested repository metadata

**Description**

Windows desktop app for color dithering, pixel-art effects, transparency control, and batch PNG export.

**Topics**

image-processing, dithering, pixel-art, python, pyside6, pillow, numpy, windows, pyinstaller

## Before making the repository public

- Confirm that the GitHub account and repository owner are the intended rights holder or have permission to publish this code.
- Review the complete Git history and working tree for credentials, personal paths, private images, and unrelated files.
- Confirm that the supplied artwork and every third-party asset may be published.
- Capture real application screenshots and add them under assets/screenshots.
- Ensure the generated assets/app_icon.ico exists when building an installer. It is ignored by Git and can be recreated from assets/app_icon_source.png with tools/generate_icon.py.
- Review COPYRIGHT.md. It reserves rights and does not provide an open-source license or a permission to use the project.
- Do not select an open-source license template unless the rights holder has decided to grant those permissions.
- Build on Windows and review the resulting portable folder or installer before attaching it to a release.
- The installer is not digitally signed; communicate that accurately in release notes.

## GitHub settings

Add the description and topics above in the repository's About settings. Keep the repository license field unset while the project has no open-source license. Add release artifacts only after a real build has been reviewed. Use a stable release tag matching the app version, such as v0.11.0-beta, only when publishing that release.

This checklist does not publish the repository, create a GitHub remote, or push commits.

# How to Get a DOI for Your GitHub Repository via Zenodo

**Zenodo** is a free, open-access repository that provides DOIs (Digital Object Identifiers) for research software, datasets, and publications. By linking your GitHub repository to Zenodo, you can automatically receive a DOI every time you create a new release.

---

## Why You Need a DOI

- A DOI makes your code **citable** and **trackable**
- Journals like **The Visual Computer** often require a DOI for the code repository
- DOIs ensure **persistent access** to your code even if the GitHub URL changes
- Zenodo is hosted by CERN — it is **free** and has **no data limits**

---

## Step-by-Step Guide

### Prerequisites

- A **GitHub account**: https://github.com
- A **Zenodo account**: https://zenodo.org (sign up with your GitHub account)
- At least **one GitHub release** created for your repository

---

### Step 1: Create a GitHub Release

Before Zenodo can generate a DOI, you need to create a tagged release on GitHub.

1. Go to your GitHub repository: `https://github.com/YitengGuo/SSDAL-Net`
2. Click **"Releases"** on the right sidebar
3. Click **"Draft a new release"**
4. Fill in the details:
   - **Tag version**: e.g., `v1.0.0`
   - **Release title**: `SSDALNet v1.0.0`
   - **Description**: Brief description of this release
5. Click **"Publish release"**

> ⚠️ **Important**: After creating the release, **do not click "Generate DOI" immediately** on Zenodo. The first time you link GitHub to Zenodo, you need to enable the integration first (Step 2).

---

### Step 2: Link GitHub to Zenodo

1. Go to **Zenodo**: https://zenodo.org
2. Log in with your GitHub account
3. Click your **profile icon** → **"GitHub"**
4. Find the **"Connect GitHub Account"** button and click it (if not already connected)
5. Once connected, scroll down to **"Sync"** section
6. Find your repository **`SSDAL-Net`** and toggle the **ON** switch to enable it
7. Zenodo will automatically mirror every GitHub release

---

### Step 3: Create Your First Zenodo Release (Manual Method)

If you prefer manual control, or if automatic sync doesn't trigger:

1. Go to **Zenodo**: https://zenodo.org
2. Click **"Upload"** on the top menu
3. Select **"GitHub"** as the upload type
4. Find and select **`YitengGuo/SSDAL-Net`**
5. Choose the version/release you want to archive
6. Fill in metadata:
   - **Title**: `SSDALNet: Synergistic Sparse-Deformable Attention Learning Network`
   - **Authors**: Yiteng Guo, Ru Xu, Hao Li, Zhibin Hao, Ye Zhang
   - **Description**: Copy from your README
   - **License**: Apache-2.0
   - **Keywords**: underwater object detection, sparse attention, deformable convolution, etc.
   - **Related identifiers**: Your paper DOI (once published)
7. Click **"Publish"**
8. Zenodo will immediately give you a **DOI badge** (e.g., `10.5281/zenodo.xxxxxx`)

---

### Step 4: Add the DOI Badge to Your README

Once you have the DOI, update your `README.md` and `CITATION.cff`:

#### Option A: Markdown Badge

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxx)
```

#### Option B: HTML Badge

```html
<a href="https://doi.org/10.5281/zenodo.xxxxxx">
  <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg" alt="DOI">
</a>
```

Add this badge to the top of your `README.md` file.

---

## Automatic vs Manual DOI

| Method | Trigger | Pros | Cons |
|--------|---------|------|------|
| **Auto-sync** | GitHub release created | Zero effort, always up-to-date | Slight delay |
| **Manual** | Zenodo upload | Full control, can add metadata | Requires manual work each release |

---

## After Your Paper Is Accepted

Once your paper in **The Visual Computer** is accepted:

1. Update the `CITATION.cff` with your **paper DOI**
2. Update the Zenodo record with the **Related Identifier** (paper DOI)
3. Create a new GitHub release (e.g., `v1.0.0-paper`)
4. This will automatically give you a **new DOI** for the paper-accepted version

---

## Troubleshooting

### "Zenodo doesn't see my GitHub release"
- Make sure you toggled **ON** for the repository in Zenodo GitHub settings
- Wait a few minutes — sync is not instant
- If still not working, use the **manual upload** method (Step 3)

### "DOI badge not showing"
- Use the format: `https://doi.org/10.5281/zenodo.xxxxxx`
- Replace `xxxxxx` with your actual Zenodo record number

### "Need to update the code after submission"
- Create a new GitHub release → Zenodo automatically creates a **new version**
- Each version gets its own DOI, but you can link them with `related_identifiers`

---

## Summary Checklist

- [ ] Create GitHub account
- [ ] Create Zenodo account and connect to GitHub
- [ ] Enable auto-sync for SSDAL-Net repository
- [ ] Create first GitHub release (`v1.0.0`)
- [ ] Wait for Zenodo to generate DOI
- [ ] Copy DOI to README badge and CITATION.cff
- [ ] After paper acceptance: add paper DOI to Zenodo record

---

## Useful Links

| Resource | URL |
|----------|-----|
| Zenodo | https://zenodo.org |
| GitHub-Zenodo Integration | https://zenodo.org/account/settings/github/ |
| DOI Format Guide | https://doi.org |
| Citation Guide | https://citation.jsession.org |

---

*Last updated: 2026-04-19*

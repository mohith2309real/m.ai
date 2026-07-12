# m.ai website — login + chat history

`site/index.html` is a full claude.ai-style product in one file: Google/email
login, per-user chat history synced in the cloud, streaming chat against your
potato server. It needs two free backends wired in.

```
site/index.html  ──login+history──▶  Firebase (Auth + Firestore)   [free, always up]
       │
       └──────────chat───────────▶  mai_serve on HF Space          [free, sleeps idle]
```

You fill in **4 Firebase values + 1 Space URL** at the top of `site/index.html`,
set 3 things in the Firebase console, add one env var on the Space, and deploy
the file. That's it.

---

## 1. Firebase (login + saved chats) — free, no card

1. **console.firebase.google.com** → Add project (or reuse your JARVIS one).
2. **Build → Authentication → Get started** → enable **Google** and
   **Email/Password** sign-in.
3. **Build → Firestore Database → Create** (production mode is fine — rules below).
4. **Project settings (⚙) → Your apps → Web (`</>`)** → register an app →
   copy the `firebaseConfig` values.
5. Paste `apiKey`, `authDomain`, `projectId`, `appId` into the
   `window.FIREBASE_CONFIG` block at the top of `site/index.html`.
   (These are *meant* to live in client code — they are not secrets.)

**Firestore security rules** — paste into Firestore → Rules → Publish. This is
what stops one user reading another's chats:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{uid}/chats/{chat} {
      allow read, write: if request.auth != null && request.auth.uid == uid;
    }
  }
}
```

---

## 2. The potato server on a Hugging Face Space

Follow `deploy/DEPLOY.md` §2 to get `mai_serve` running on a Space. Then the
one extra step the API-only setup didn't need — **CORS**, because the website
and the Space are different origins:

- HF Space → **Settings → Variables and secrets** → add a **variable**
  `MAI_ORIGINS` = your site origin, e.g. `https://your-project.web.app`
  (or `*` to allow any page — fine for an open demo).

Put the same Space URL into `window.DEFAULT_API` at the top of
`site/index.html` (users can also change it in the site's ⚙ Settings).

> Honest note: Firebase login gates the **website UI**, but the Space's API
> stays publicly reachable (anyone with the URL can POST to it). For a potato
> demo that's fine. Real per-user API gating would need a paid always-on proxy.

---

## 3. Deploy the website — free, always up

**Option A — Firebase Hosting** (same project, one command):
```bash
npm i -g firebase-tools
cd site && firebase init hosting   # pick your project, public dir = "."
firebase deploy
```
Live at `https://your-project.web.app`.

**Option B — GitHub Pages** (zero install): drop `site/index.html` in a repo,
Settings → Pages → deploy from branch. Live at `https://you.github.io/repo/`.

**After deploying:** Firebase console → Authentication → Settings →
**Authorized domains** → add your live domain, or Google sign-in will refuse.

---

## Local preview

```bash
python3 -m http.server 8130 --directory site   # then open http://localhost:8130
```
Login won't work until Firebase is filled in, but you'll see the UI. Add
`http://localhost` to Firebase authorized domains to test login locally.

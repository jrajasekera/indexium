# Future Ideas

## Continuous Library Watcher
Set up a background watcher that notices when new or changed video files arrive in the configured directories, automatically schedules them for hashing, and surfaces their status in the UI so users never have to remember to rerun the scanner manually.

## Guided Auto-Labeling With Confidence Feedback
Introduce an active-learning workflow where the system records similarity scores for suggested names, shows the confidence to the reviewer, and asks for quick thumbs-up/thumbs-down feedback to continuously improve future automatic assignments.

## Tagging Progress & Quality Dashboard
Create a dashboard that highlights how many clusters remain unnamed, which people have the highest number of unreviewed faces, and trends over time so users can focus their effort where it matters most.

## Contextual Video Preview Snippets
Allow reviewers to launch a short clip around any thumbnail directly from the browser, giving them motion and audio context that helps disambiguate similar-looking faces or twins.

## Recovery Center for Failed Videos
Add a dedicated page that lists every video stuck in the failed status, displays the collected error messages, and offers one-click retry or manual resolution guidance.

## Collaborative Tagging & Audit Trail
Support multiple reviewers by adding lightweight authentication, per-user assignment queues, and an audit history that logs who named, renamed, or deleted each face.

## Face Quality & Deduplication Tools
Flag faces with low sharpness, heavy occlusion, or near-duplicate thumbnails, then let users bulk delete or reassign them to keep the dataset clean.

## Scalable Similarity Search Backend
Back the face-encoding lookups with a dedicated similarity index that scales to hundreds of thousands of faces while keeping merge suggestions and clustering operations fast.

## Smart Metadata Writing Planner
Before writing metadata, show a preview of which files will change, what tags will be added, and whether existing comments will be overwritten so the user can confirm or postpone individual items.

# RescueGrid frontend

This is a dependency-free static client for Vercel. The existing Python code stays in app/ and src/; this folder talks to the Render API in backend/.

## Deploy to Vercel

1. Import the RescueGrid GitHub repository in Vercel.
2. Set **Root Directory** to frontend.
3. Choose **Other** as the framework preset. Leave the build and install commands blank and set the output directory to the project root.
4. Deploy. Open the site, paste the Render service URL into **Render API URL**, then select **Check**.

The API URL is saved in browser local storage. It is not a build-time secret or a credential.

## Run locally

Start the backend from the repository root with uvicorn backend.main:app --reload --port 8000. In another terminal run python -m http.server 3000 --directory frontend, then open http://localhost:3000. The local API is prefilled in the form.

The form accepts a WGS84 bounding box (west, south, east, north) and event date. Each bounding-box side is limited to 0.25 degrees. Click the map to fill a focused 0.04-degree area.

Analysis jobs and run artifacts are held on the API service's local disk and in process memory. A Render restart clears job state and may remove run artifacts; submit the analysis again if a job is no longer available.

.headers on
.mode column

SELECT COUNT(*) AS scanned_files FROM scanned_files;
SELECT processing_status, COUNT(*) AS count FROM scanned_files GROUP BY processing_status;
SELECT manual_review_status, COUNT(*) AS count FROM scanned_files GROUP BY manual_review_status;

SELECT COUNT(*) AS faces FROM faces;
SELECT CASE WHEN cluster_id IS NULL THEN 'unclustered' ELSE 'clustered' END AS status,
       COUNT(*) AS count
FROM faces
GROUP BY status;

SELECT COUNT(DISTINCT cluster_id) AS unnamed_clusters
FROM faces
WHERE cluster_id IS NOT NULL AND person_name IS NULL;

SELECT COUNT(*) AS named_faces
FROM faces
WHERE person_name IS NOT NULL;

SELECT COUNT(DISTINCT sf.file_hash) AS videos_with_faces_no_clusters
FROM scanned_files sf
WHERE sf.face_count > 0
  AND NOT EXISTS (
    SELECT 1 FROM faces f
    WHERE f.file_hash = sf.file_hash AND f.cluster_id IS NOT NULL
  );

SELECT error_message, COUNT(*) AS count
FROM scanned_files
WHERE processing_status = 'failed'
GROUP BY error_message
ORDER BY count DESC
LIMIT 10;

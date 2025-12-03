# Deployment Testing Guide

This document describes how to test the Face Attribute Classification API deployment.

## Prerequisites for Testing

- Docker and Docker Compose installed
- At least 5GB of free disk space (for Docker images and dependencies)
- Sample face images for testing

## Test 1: Docker Build

Test that the Docker image builds successfully:

```bash
docker build -t face-attribute-api:test .
```

**Expected Result:** Image builds without errors.

## Test 2: Docker Compose Validation

Validate the docker-compose configuration:

```bash
docker compose config
```

**Expected Result:** Configuration is displayed without errors.

## Test 3: Container Startup

Start the container using docker-compose:

```bash
docker compose up
```

**Expected Result:** 
- Container starts successfully
- API server starts on port 8000
- Logs show "API started successfully!"
- No error messages in startup

## Test 4: Health Check

Test the health check endpoint:

```bash
curl http://localhost:8000/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "data_dir": "/app/data",
  "images_dir_exists": true,
  "annotations_exists": false
}
```

## Test 5: Web Interface Access

Open http://localhost:8000 in a web browser.

**Expected Result:**
- Web page loads successfully
- Attribute selection grid is displayed
- All 38 attributes are shown
- UI is responsive and interactive

## Test 6: API Documentation

Open http://localhost:8000/docs in a web browser.

**Expected Result:**
- Swagger/OpenAPI documentation is displayed
- All endpoints are listed:
  - GET /
  - GET /api/attributes
  - POST /api/search
  - GET /api/health

## Test 7: Get Attributes Endpoint

Test the attributes listing endpoint:

```bash
curl http://localhost:8000/api/attributes
```

**Expected Response:**
```json
{
  "attributes": ["Male", "Young", "Smiling", ...]
}
```

## Test 8: Search Endpoint (Without Images)

Test the search endpoint when no images are present:

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "attributes": ["Male", "Smiling"],
    "threshold": 0.5,
    "limit": 10
  }'
```

**Expected Response:**
```json
{
  "query": {
    "attributes": ["Male", "Smiling"],
    "threshold": 0.5,
    "limit": 10
  },
  "results": [],
  "count": 0
}
```

## Test 9: Volume Mount Verification

Verify that the volume mount works:

```bash
# Create a test file in the data directory
echo "test" > data/test.txt

# Check if it's visible in the container
docker compose exec face-attribute-api ls -la /app/data/

# Clean up
rm data/test.txt
```

**Expected Result:** File is visible inside the container at /app/data/test.txt

## Test 10: Search with Images

1. Add some test images to `data/images/`:
   ```bash
   mkdir -p data/images
   # Copy some face images to data/images/
   ```

2. Restart the container:
   ```bash
   docker compose restart
   ```

3. Search using the web interface:
   - Open http://localhost:8000
   - Select attributes (e.g., "Male", "Smiling")
   - Click "Search Images"

**Expected Result:** 
- Search completes without errors
- Matching images are displayed (if any match the criteria)
- Each image shows its filename, match score, and attribute values

## Test 11: API Search with Images

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "attributes": ["Male"],
    "threshold": 0.5,
    "limit": 5
  }'
```

**Expected Response:**
```json
{
  "query": {...},
  "results": [
    {
      "filename": "image001.jpg",
      "path": "/images/image001.jpg",
      "attributes": {
        "Male": 0.95
      },
      "match_score": 0.95
    }
  ],
  "count": 1
}
```

## Test 12: Pre-computed Annotations

1. Create an annotations CSV at `data/annotations.csv`:
   ```csv
   filename,Male,Young,Smiling,...
   image1.jpg,0.95,0.87,0.92,...
   ```

2. Restart the container:
   ```bash
   docker compose restart
   ```

3. Check logs:
   ```bash
   docker compose logs face-attribute-api
   ```

**Expected Result:** Logs show "Loading annotations from /app/data/annotations.csv"

## Test 13: Error Handling

Test invalid attribute names:

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "attributes": ["InvalidAttribute"],
    "threshold": 0.5,
    "limit": 10
  }'
```

**Expected Response:** 400 error with message about invalid attributes

## Test 14: Container Restart

Test that the container restarts properly:

```bash
docker compose restart
```

**Expected Result:** Container restarts successfully and API is accessible

## Test 15: Container Logs

Check container logs for errors:

```bash
docker compose logs face-attribute-api
```

**Expected Result:** No error or warning messages (except expected ones like "No annotations file found")

## Test 16: Cleanup

Stop and remove containers:

```bash
docker compose down
```

**Expected Result:** Containers are stopped and removed cleanly

## Performance Tests

### Test P1: Search Performance

Measure search time with different numbers of images:
- 10 images
- 100 images
- 1000 images

**Expected Result:** 
- Search completes in reasonable time
- Performance degrades gracefully with more images
- Pre-computed annotations provide faster search

### Test P2: Concurrent Requests

Use tools like Apache Bench or wrk to test concurrent requests:

```bash
ab -n 100 -c 10 http://localhost:8000/api/health
```

**Expected Result:** API handles concurrent requests without errors

## Security Tests

### Test S1: Input Validation

Test with various invalid inputs:
- Negative threshold
- Threshold > 1
- Negative limit
- Extremely large limit

**Expected Result:** API returns appropriate error responses

### Test S2: XSS Prevention

Test that the frontend properly handles special characters in filenames and paths.

**Expected Result:** No XSS vulnerabilities in the web interface

## Integration Test Checklist

- [ ] Docker image builds successfully
- [ ] Container starts without errors
- [ ] Health check endpoint returns healthy status
- [ ] Web interface loads and displays attributes
- [ ] API documentation is accessible
- [ ] Volume mount works correctly
- [ ] Search works with and without images
- [ ] Pre-computed annotations are loaded correctly
- [ ] Error handling works properly
- [ ] Container can be restarted
- [ ] No security vulnerabilities

## Troubleshooting Common Issues

### Issue: Container fails to start
**Solution:** Check logs with `docker compose logs` and verify all files are present

### Issue: Images not found
**Solution:** Ensure images are in `data/images/` and volume mount is correct

### Issue: Slow search
**Solution:** Use pre-computed annotations or enable GPU support

### Issue: Out of memory
**Solution:** Reduce `MAX_IMAGES_TO_PROCESS` or allocate more memory to Docker

## CI/CD Integration

For automated testing in CI/CD pipelines:

```bash
# Build
docker build -t face-attribute-api:test .

# Run tests
docker compose up -d
sleep 10  # Wait for startup
curl -f http://localhost:8000/api/health || exit 1
docker compose down

# Push to registry
docker tag face-attribute-api:test your-registry/face-attribute-api:latest
docker push your-registry/face-attribute-api:latest
```

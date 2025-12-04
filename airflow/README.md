# Airflow Orchestration for ML Pipeline

This directory contains the Airflow setup for orchestrating the ML pipeline using DVC.

## 📋 Overview

The ML pipeline consists of the following stages:
1. **prepare_data** - Prepare and preprocess training data
2. **hyperopt_search** - Optimize hyperparameters using Hyperopt
3. **train_model** - Train the model with best hyperparameters
4. **evaluate_model** - Evaluate model on test set
5. **batch_inference** - Generate predictions for unlabeled data
6. **notify_completion** - Send notification when pipeline completes

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB of RAM available for Docker

### 1. Start Airflow

```bash
cd airflow
docker-compose -f docker-compose.airflow.yml up -d
```

This will start:
- **Airflow Webserver** on http://localhost:8080
- **Airflow Scheduler** for task execution
- **PostgreSQL** for Airflow metadata

### 2. Access Airflow UI

1. Open your browser and go to: http://localhost:8080
2. Login with default credentials:
   - **Username**: `airflow`
   - **Password**: `airflow`

### 3. Trigger the Pipeline

#### Option 1: Full Pipeline (with Hyperopt)
- In the Airflow UI, find the `ml_pipeline` DAG
- Toggle it to "ON" if it's paused
- Click the "Play" button to trigger a manual run

#### Option 2: Retrain Only (skip Hyperopt)
- Use the `ml_retrain_only` DAG
- This reuses existing hyperparameters and is faster

#### Option 3: Inference Only
- Use the `ml_inference_only` DAG
- Runs daily automatically to generate new predictions

## 📊 Available DAGs

### 1. ml_pipeline (Main DAG)
- **Schedule**: Weekly (`@weekly`)
- **Description**: Complete ML workflow including hyperparameter optimization
- **Tasks**: prepare_data → hyperopt → train → [evaluate, inference] → notify

### 2. ml_retrain_only
- **Schedule**: Manual trigger
- **Description**: Quick retrain without hyperparameter optimization
- **Tasks**: train → [evaluate, inference] → notify

### 3. ml_inference_only
- **Schedule**: Daily (`@daily`)
- **Description**: Run inference on new data only
- **Tasks**: batch_inference

## 🔧 Configuration

### Environment Variables

You can customize the setup by setting environment variables:

```bash
# In airflow/.env file (create if it doesn't exist)
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=your_secure_password
```

### Project Path

The project is mounted at `/opt/airflow/project` inside the container. If you need to change this, edit `docker-compose.airflow.yml`:

```yaml
volumes:
  - ../:/opt/airflow/project  # Change this path if needed
```

### DAG Configuration

Edit `dags/ml_pipeline_dag.py` to customize:
- **Schedule intervals**: Change `schedule_interval` parameter
- **Retry policy**: Modify `retries` and `retry_delay` in `default_args`
- **Email notifications**: Update `email` in `default_args`
- **Task dependencies**: Modify task dependencies at the end of the DAG definition

## 📝 Monitoring

### Check DAG Status
```bash
# View running containers
docker-compose -f docker-compose.airflow.yml ps

# View scheduler logs
docker-compose -f docker-compose.airflow.yml logs -f airflow-scheduler

# View webserver logs
docker-compose -f docker-compose.airflow.yml logs -f airflow-webserver
```

### Access Logs
Logs are stored in `airflow/logs/` directory and can be viewed:
- In the Airflow UI (Task Instance → Log)
- Directly in the filesystem

## 🛠️ Troubleshooting

### Issue: "service 'airflow-init' didn't complete successfully: exit 1"

This is usually caused by permission issues or database initialization problems.

**Solutions:**

1. **Set correct AIRFLOW_UID** (recommended):
   ```bash
   # Get your user ID
   echo $UID
   
   # Set it before starting (Linux/Mac)
   export AIRFLOW_UID=$(id -u)
   docker-compose -f docker-compose.airflow.yml up -d
   ```

2. **Create required directories with correct permissions**:
   ```bash
   cd airflow
   mkdir -p logs dags plugins config
   chmod -R 777 logs plugins config  # Or set to your user
   ```

3. **Clean start** (if issues persist):
   ```bash
   # Stop and remove everything
   docker-compose -f docker-compose.airflow.yml down -v
   
   # Clean directories
   rm -rf logs/* plugins/* config/*
   
   # Restart with correct UID
   export AIRFLOW_UID=$(id -u)
   docker-compose -f docker-compose.airflow.yml up -d
   ```

4. **Check database initialization logs**:
   ```bash
   docker-compose -f docker-compose.airflow.yml logs airflow-init
   ```

### Issue: DAGs not showing up
- Check that DAG files are in `airflow/dags/` directory
- Check scheduler logs: `docker-compose -f docker-compose.airflow.yml logs airflow-scheduler`
- Verify DAG syntax: The scheduler will log any Python errors

### Issue: Tasks failing with "command not found"
- Ensure required Python packages are installed in the Airflow image
- Check `_PIP_ADDITIONAL_REQUIREMENTS` in `docker-compose.airflow.yml`

### Issue: Permission errors
- Check that `AIRFLOW_UID` matches your user ID
- Run: `echo $UID` to check your user ID
- Set it in docker-compose: `AIRFLOW_UID=<your-uid>`

### Issue: Cannot connect to PostgreSQL
- Wait for PostgreSQL to be fully initialized (may take 30-60 seconds on first start)
- Check health status: `docker-compose -f docker-compose.airflow.yml ps`

## 🔄 Updating the Pipeline

After modifying DAG files:
1. Wait for Airflow to auto-reload (happens every few seconds)
2. Or restart the scheduler: `docker-compose -f docker-compose.airflow.yml restart airflow-scheduler`

## 🛑 Stopping Airflow

```bash
# Stop all services
docker-compose -f docker-compose.airflow.yml down

# Stop and remove volumes (clears database)
docker-compose -f docker-compose.airflow.yml down -v
```

## 📚 Additional Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)

## 🔐 Security Notes

For production use:
1. Change default passwords in `.env` file
2. Set up proper email configuration for notifications
3. Configure SSL/TLS for the webserver
4. Use secrets backend for sensitive data
5. Implement proper authentication (LDAP, OAuth, etc.)

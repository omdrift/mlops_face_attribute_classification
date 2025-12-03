"""
Tests for Airflow DAGs
"""
import pytest
from datetime import datetime
from airflow.models import DagBag


class TestAirflowDAGs:
    """Tests for Airflow DAG definitions"""
    
    @pytest.fixture(scope="class")
    def dagbag(self):
        """Load all DAGs"""
        return DagBag(dag_folder='airflow/dags', include_examples=False)
    
    def test_no_import_errors(self, dagbag):
        """Test that all DAGs can be imported without errors"""
        assert len(dagbag.import_errors) == 0, \
            f"DAG import errors: {dagbag.import_errors}"
    
    def test_ml_pipeline_dag_loaded(self, dagbag):
        """Test that ml_pipeline DAG is loaded"""
        assert 'ml_pipeline' in dagbag.dags
        dag = dagbag.get_dag('ml_pipeline')
        assert dag is not None
    
    def test_monitoring_dag_loaded(self, dagbag):
        """Test that monitoring_pipeline DAG is loaded"""
        assert 'monitoring_pipeline' in dagbag.dags
        dag = dagbag.get_dag('monitoring_pipeline')
        assert dag is not None
    
    def test_retraining_dag_loaded(self, dagbag):
        """Test that retraining_pipeline DAG is loaded"""
        assert 'retraining_pipeline' in dagbag.dags
        dag = dagbag.get_dag('retraining_pipeline')
        assert dag is not None
    
    def test_ml_pipeline_structure(self, dagbag):
        """Test ml_pipeline DAG structure"""
        dag = dagbag.get_dag('ml_pipeline')
        
        # Check tasks exist
        expected_tasks = [
            'check_new_data',
            'preprocess_data',
            'train_model',
            'evaluate_model',
            'check_drift',
            'check_deployment_criteria'
        ]
        
        task_ids = [task.task_id for task in dag.tasks]
        for expected_task in expected_tasks:
            assert expected_task in task_ids, \
                f"Task '{expected_task}' not found in ml_pipeline"
    
    def test_monitoring_pipeline_structure(self, dagbag):
        """Test monitoring_pipeline DAG structure"""
        dag = dagbag.get_dag('monitoring_pipeline')
        
        # Check tasks exist
        expected_tasks = [
            'collect_inference_metrics',
            'generate_drift_report',
            'check_alerts',
            'send_notifications'
        ]
        
        task_ids = [task.task_id for task in dag.tasks]
        for expected_task in expected_tasks:
            assert expected_task in task_ids, \
                f"Task '{expected_task}' not found in monitoring_pipeline"
    
    def test_retraining_pipeline_structure(self, dagbag):
        """Test retraining_pipeline DAG structure"""
        dag = dagbag.get_dag('retraining_pipeline')
        
        # Check tasks exist
        expected_tasks = [
            'check_retraining_criteria',
            'trigger_retraining',
            'run_dvc_pipeline',
            'train_model'
        ]
        
        task_ids = [task.task_id for task in dag.tasks]
        for expected_task in expected_tasks:
            assert expected_task in task_ids, \
                f"Task '{expected_task}' not found in retraining_pipeline"
    
    def test_dag_schedules(self, dagbag):
        """Test that DAGs have correct schedules"""
        ml_dag = dagbag.get_dag('ml_pipeline')
        monitoring_dag = dagbag.get_dag('monitoring_pipeline')
        retraining_dag = dagbag.get_dag('retraining_pipeline')
        
        assert ml_dag.schedule_interval == '@weekly'
        assert monitoring_dag.schedule_interval == '@daily'
        assert retraining_dag.schedule_interval == '0 2 * * 1'  # Monday 2 AM
    
    def test_dag_default_args(self, dagbag):
        """Test that DAGs have proper default args"""
        ml_dag = dagbag.get_dag('ml_pipeline')
        
        assert ml_dag.default_args.get('owner') == 'mlops-team'
        assert ml_dag.default_args.get('retries') is not None
        assert ml_dag.default_args.get('email_on_failure') is True
    
    def test_task_dependencies_ml_pipeline(self, dagbag):
        """Test task dependencies in ml_pipeline"""
        dag = dagbag.get_dag('ml_pipeline')
        
        # Get tasks
        check_data = dag.get_task('check_new_data')
        preprocess = dag.get_task('preprocess_data')
        train = dag.get_task('train_model')
        
        # Check dependencies
        assert preprocess in check_data.downstream_list
        assert train in preprocess.downstream_list
    
    def test_task_dependencies_monitoring_pipeline(self, dagbag):
        """Test task dependencies in monitoring_pipeline"""
        dag = dagbag.get_dag('monitoring_pipeline')
        
        # Get tasks
        collect = dag.get_task('collect_inference_metrics')
        generate = dag.get_task('generate_drift_report')
        check = dag.get_task('check_alerts')
        notify = dag.get_task('send_notifications')
        
        # Check dependencies form a linear chain
        assert generate in collect.downstream_list
        assert check in generate.downstream_list
        assert notify in check.downstream_list


class TestDAGTasks:
    """Tests for individual DAG tasks"""
    
    def test_check_new_data_function(self):
        """Test check_new_data function logic"""
        from airflow.dags.ml_pipeline_dag import check_new_data
        
        # Mock context
        context = {}
        
        # Should return boolean
        result = check_new_data(**context)
        assert isinstance(result, bool)
    
    def test_check_model_metrics_function(self):
        """Test check_model_metrics function logic"""
        from airflow.dags.ml_pipeline_dag import check_model_metrics
        
        context = {}
        
        # Should return task_id to execute
        result = check_model_metrics(**context)
        assert result in ['deploy_model', 'skip_deploy']

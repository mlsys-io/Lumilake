import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server.utils.job_storage import InMemoryJobStorage

job_storage_module._job_storage = InMemoryJobStorage()

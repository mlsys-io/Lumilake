import lumilake.utils.job_storage as job_storage_module
from lumilake.utils.job_storage import InMemoryJobStorage

job_storage_module._job_storage = InMemoryJobStorage()

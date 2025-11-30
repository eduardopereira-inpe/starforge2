import os
import json
from typing import Optional
from numpy import array, ndarray
from pydal import DAL, Field

from .massfunctionschema import SigmaCache


class SigmaCacheManager:
    """
    Handle persistence of σ(M, z) computations.
    Provides load/save functionality with SQLite backend.
    """

    _db_instance = None  # singleton DB

    @classmethod
    def get_db(cls, db_path: Optional[str] = None):
        """Return singleton DAL database instance."""
        if cls._db_instance is None:
            if db_path is None:
                home = os.path.expanduser("~")
                cache_dir = os.path.join(home, ".starforge")
                os.makedirs(cache_dir, exist_ok=True)
                db_path = os.path.join(cache_dir, "sigma_cache.sqlite")

            cls._db_instance = DAL(f"sqlite://{db_path}")
            cls._db_instance.define_table(
                "sigma_data",
                Field("name", "string", unique=True),
                Field("data", "text"),
            )
            cls._db_instance.commit()
        return cls._db_instance

    def __init__(self, db_path: Optional[str] = None):
        self.db = self.get_db(db_path)

    def load(self, name: str) -> Optional[SigmaCache]:
        """Load a SigmaCache by name from DB."""
        record = self.db(self.db.sigma_data.name == name).select().first()
        if record:
            data = json.loads(record.data)
            for k, v in data.items():
                if isinstance(v, list):
                    data[k] = array(v)
            return SigmaCache(**data)
        return None

    def save(self, name: str, sigma_cache: SigmaCache):
        """Save a SigmaCache object to DB."""
        data_dict = sigma_cache.model_dump()
        for k, v in data_dict.items():
            if isinstance(v, ndarray):
                data_dict[k] = v.tolist()
        record = self.db(self.db.sigma_data.name == name).select().first()
        if record:
            record.update_record(data=json.dumps(data_dict))
        else:
            self.db.sigma_data.insert(name=name, data=json.dumps(data_dict))
        self.db.commit()

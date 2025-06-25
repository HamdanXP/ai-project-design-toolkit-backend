from fastapi import APIRouter
from api.v1 import projects, reflection, scoping, development, evaluation, admin

api_router = APIRouter()

api_router.include_router(projects.router)
api_router.include_router(reflection.router)
api_router.include_router(scoping.router)
api_router.include_router(development.router)
api_router.include_router(evaluation.router)
api_router.include_router(admin.router)

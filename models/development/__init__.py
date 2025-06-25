"""Expose development related models and enums.

This package provides a small wrapper around the main ``models/development.py``
module so that other parts of the code base can simply import from
``models.development`` without worrying whether they're referring to the module
or the package.  The previous implementation only exported the enum values which
meant that imports such as ``from models.development import ProjectContext``
failed because ``ProjectContext`` is defined in ``models/development.py`` and was
never re-exported here.  To make the API consistent we re-export the commonly
used classes from the module.
"""

from .enums import AITechnique, DeploymentStrategy, ComplexityLevel
from ..development import (
    ResourceRequirement,
    ProjectRecommendation,
    EthicalSafeguard,
    TechnicalArchitecture,
    AISolution,
    ProjectContext,
    ProjectContextOnly,
    SolutionsData,
    DevelopmentPhaseData,
    GeneratedProject,
    ProjectGenerationRequest,
    ProjectGenerationResponse,
)

__all__ = [
    "AITechnique",
    "DeploymentStrategy",
    "ComplexityLevel",
    "ResourceRequirement",
    "ProjectRecommendation",
    "EthicalSafeguard",
    "TechnicalArchitecture",
    "AISolution",
    "ProjectContext",
    "ProjectContextOnly",
    "SolutionsData",
    "DevelopmentPhaseData",
    "GeneratedProject",
    "ProjectGenerationRequest",
    "ProjectGenerationResponse",
]

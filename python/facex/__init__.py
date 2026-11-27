"""
FaceX — Python binding for face detection + recognition.

Usage:
    from facex import FaceX
    fx = FaceX()
    faces = fx.detect(image_rgb)       # numpy uint8 HWC
    sim = FaceX.similarity(emb1, emb2) # cosine similarity
"""

from facex.core import FaceX, similarity

from facex.gallery import CompactGallery, Gallery, enrollment_template
from facex.identity import IdentityGallery

__version__ = "0.1.0"
__all__ = ["FaceX", "IdentityGallery", "CompactGallery", "Gallery", "enrollment_template", "similarity"]

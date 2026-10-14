"""CPU-only checks of immutable launcher image selection."""
from start_tp2 import immutable_image_id
h='a'*64
assert immutable_image_id('sha256:'+h)=='sha256:'+h
assert immutable_image_id('device0/model@sha256:'+h) is None
for value in ['latest','model:tag','sha256:abc','sha256:'+'G'*64,'repo@sha256:short','a b@sha256:'+h]:
    try:immutable_image_id(value)
    except ValueError:pass
    else:raise AssertionError(value)
print('IMMUTABLE_IMAGE_SELECTION_PASS')

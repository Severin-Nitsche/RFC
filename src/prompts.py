import config

from string import Template

annotate = None
classify = None
verify = None

with open(config.ANNOTATE, 'r') as annotate_file:
    annotate = Template(annotate_file.read())

with open(config.CLASSIFY, 'r') as classify_file:
    classify = Template(classify_file.read())

with open(config.VERIFY, 'r') as verify_file:
    verify = Template(verify_file.read())
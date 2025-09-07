from apiflask import Schema, fields
from marshmallow import validate
from marshmallow.exceptions import ValidationError
from marshmallow.decorators import validates_schema
from marshmallow.validate import Length


class StringOrList(fields.Field):
    def __init__(self, **kwargs):
        metadata = {
            'type': ['string', 'array'],
            'items': {'type': 'string'},
        }

        if 'metadata' in kwargs:
            kwargs['metadata'] = {**metadata, **kwargs['metadata']}
        else:
            kwargs['metadata'] = metadata

        super().__init__(**kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, str):
            return value
        elif isinstance(value, list) and all(isinstance(i, str) for i in value):
            return value
        raise ValidationError("Must be a string or a list of strings")

    def _serialize(self, value, attr, obj, **kwargs):
        return value


class ModelStats(Schema):
    tasks = fields.List(fields.String())
    name = fields.String()
    locale = fields.List(fields.String())
    url = fields.String()
    size = fields.Number(metadata={
        'description': 'Size of the model in bytes, if no files are found it is `0`.'
    })


class ModelsResponse(Schema):
    total = fields.Number()
    stats = fields.Dict(fields.String(), fields.Nested(ModelStats()))


class InferUsageStats(Schema):
    stage = fields.String()
    dur = fields.Number()
    tokens = fields.Number()


class InferUsage(Schema):
    model = fields.String()
    stats = fields.List(fields.Nested(InferUsageStats()))


class InferBaseResponse(Schema):
    usage = fields.List(fields.Nested(InferUsage()))


class OutcomeTextPairsResponse(InferBaseResponse):
    # todo: should be tuple https://github.com/marshmallow-code/apispec/issues/399
    # outcome = fields.List(
    #     fields.Tuple((fields.String(), fields.String())),
    #     # metadata={'x-widget': 'GenericList'},
    #     metadata={
    #         'widget': 'GenericList',
    #         # 'items': {
    #         #     'type': 'array',
    #         #     # when moving `items` inside metadata of `Tuple` the "unhashable type: 'list'" error is thrown
    #         #     'items': [{'type': 'string'}, {'type': 'string'}],
    #         # },
    #     },
    # )
    outcome = fields.List(fields.List(fields.String()))


class OutcomeWithScoresResponse(InferBaseResponse):
    outcome = fields.List(fields.Number())

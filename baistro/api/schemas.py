from apiflask import Schema, fields


class ModelStats(Schema):
    tasks = fields.List(fields.String())
    name = fields.String()
    locale = fields.List(fields.String())
    url = fields.String()
    size = fields.Number()


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
    _usages = fields.List(fields.Nested(InferUsage()))


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


class VectorRequest(Schema):
    input = fields.String()


class VectorBatchRequest(Schema):
    input = fields.List(fields.String())


class VectorQueryRequest(Schema):
    options = fields.Dict()
    # todo: implement batch query, if not just broken with schema
    query = fields.String()
    context = fields.List(fields.String())


class VectorFileRequest(Schema):
    file = fields.File()
    input = fields.String()


class VectorResponse(InferBaseResponse):
    embeddings = fields.List(fields.Number())


class VectorBatchResponse(InferBaseResponse):
    embeddings = fields.List(fields.List(fields.Number()))


class VectorQueryMatch(Schema):
    match = fields.String()
    score = fields.Float()


class VectorQueryResponse(InferBaseResponse):
    matches = fields.List(fields.Nested(VectorQueryMatch()))

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: mitrailleuse.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'mitrailleuse.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12mitrailleuse.proto\x12\x0cmitrailleuse\"^\n\x11\x43reateTaskRequest\x12\x0f\n\x07user_id\x18\x01 \x01(\t\x12\x10\n\x08\x61pi_name\x18\x02 \x01(\t\x12\x11\n\ttask_name\x18\x03 \x01(\t\x12\x13\n\x0b\x63onfig_json\x18\x04 \x01(\t\")\n\x12\x43reateTaskResponse\x12\x13\n\x0btask_folder\x18\x01 \x01(\t\":\n\x12\x45xecuteTaskRequest\x12\x0f\n\x07user_id\x18\x01 \x01(\t\x12\x13\n\x0btask_folder\x18\x02 \x01(\t\"5\n\x13\x45xecuteTaskResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\x12\x0e\n\x06job_id\x18\x02 \x01(\t\"<\n\x14GetTaskStatusRequest\x12\x0f\n\x07user_id\x18\x01 \x01(\t\x12\x13\n\x0btask_folder\x18\x02 \x01(\t\"\'\n\x15GetTaskStatusResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\"6\n\x10ListTasksRequest\x12\x0f\n\x07user_id\x18\x01 \x01(\t\x12\x11\n\ttask_name\x18\x02 \x01(\t\":\n\x11ListTasksResponse\x12%\n\x05tasks\x18\x01 \x03(\x0b\x32\x16.mitrailleuse.TaskInfo\")\n\x14GetTaskByPathRequest\x12\x11\n\ttask_path\x18\x01 \x01(\t\"^\n\x08TaskInfo\x12\x0f\n\x07user_id\x18\x01 \x01(\t\x12\x10\n\x08\x61pi_name\x18\x02 \x01(\t\x12\x11\n\ttask_name\x18\x03 \x01(\t\x12\x0e\n\x06status\x18\x04 \x01(\t\x12\x0c\n\x04path\x18\x05 \x01(\t2\xb3\x03\n\x13MitrailleuseService\x12O\n\nCreateTask\x12\x1f.mitrailleuse.CreateTaskRequest\x1a .mitrailleuse.CreateTaskResponse\x12R\n\x0b\x45xecuteTask\x12 .mitrailleuse.ExecuteTaskRequest\x1a!.mitrailleuse.ExecuteTaskResponse\x12X\n\rGetTaskStatus\x12\".mitrailleuse.GetTaskStatusRequest\x1a#.mitrailleuse.GetTaskStatusResponse\x12N\n\tListTasks\x12\x1e.mitrailleuse.ListTasksRequest\x1a\x1f.mitrailleuse.ListTasksResponse\"\x00\x12M\n\rGetTaskByPath\x12\".mitrailleuse.GetTaskByPathRequest\x1a\x16.mitrailleuse.TaskInfo\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mitrailleuse_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_CREATETASKREQUEST']._serialized_start=36
  _globals['_CREATETASKREQUEST']._serialized_end=130
  _globals['_CREATETASKRESPONSE']._serialized_start=132
  _globals['_CREATETASKRESPONSE']._serialized_end=173
  _globals['_EXECUTETASKREQUEST']._serialized_start=175
  _globals['_EXECUTETASKREQUEST']._serialized_end=233
  _globals['_EXECUTETASKRESPONSE']._serialized_start=235
  _globals['_EXECUTETASKRESPONSE']._serialized_end=288
  _globals['_GETTASKSTATUSREQUEST']._serialized_start=290
  _globals['_GETTASKSTATUSREQUEST']._serialized_end=350
  _globals['_GETTASKSTATUSRESPONSE']._serialized_start=352
  _globals['_GETTASKSTATUSRESPONSE']._serialized_end=391
  _globals['_LISTTASKSREQUEST']._serialized_start=393
  _globals['_LISTTASKSREQUEST']._serialized_end=447
  _globals['_LISTTASKSRESPONSE']._serialized_start=449
  _globals['_LISTTASKSRESPONSE']._serialized_end=507
  _globals['_GETTASKBYPATHREQUEST']._serialized_start=509
  _globals['_GETTASKBYPATHREQUEST']._serialized_end=550
  _globals['_TASKINFO']._serialized_start=552
  _globals['_TASKINFO']._serialized_end=646
  _globals['_MITRAILLEUSESERVICE']._serialized_start=649
  _globals['_MITRAILLEUSESERVICE']._serialized_end=1084
# @@protoc_insertion_point(module_scope)

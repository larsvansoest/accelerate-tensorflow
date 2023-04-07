{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE InstanceSigs #-}

module Data.Accelerate.TensorFlow.Operation where

import Data.Accelerate.TensorFlow.Type
    ( TFOrd, OneOf, TFNum, TFAll, TFFloat, TensorType, TFInt, TFNum', TFMod )
import Data.Array.Accelerate.Type ( ScalarType )
import Data.Array.Accelerate.AST.Operation
    ( PrimBool, Mut, Out, In, NFData'(..), Var', ExpVar )
import Data.Array.Accelerate
    ( MakesILP(..),
      ShrinkArg(..),
      SLVOperation(..),
      SimplifyOperation,
      PrettyOp(prettyOp),
      EncodeOperation(..))
import Data.Array.Accelerate.Pretty.Exp ( prettyConst, Adoc )
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Labels (LabelledArg(..))
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Graph
    ( LabelledArgOp(LOp) )
import Data.Array.Accelerate.Analysis.Hash.Exp (intHost, hashQ, encodeScalarType, encodeScalarConst, Builder, encodeExpVar)
import Prettyprinter ( viaShow, vsep )
import Data.Array.Accelerate.Pretty.Type ( prettyScalarType )
import Data.Array.Accelerate.Representation.Type
    ( TupR(TupRsingle), TypeR )
import Data.Array.Accelerate.Representation.Shape (DIM1, ShapeR)

data TensorOp op where
  TConstant :: OneOf TFAll a => ScalarType a -> a -> TensorOp (Out sh a -> ())
  TVar      :: OneOf TFAll a => ScalarType a -> TensorOp (Var' a -> Out sh a -> ())
  TId       :: OneOf TFAll a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TSelect   :: OneOf TFAll a => ScalarType a -> TensorOp (In sh (PrimBool, (a, a)) -> Out sh a -> ())
  TGather   :: OneOf TFAll a => ScalarType a -> TensorOp (In DIM1 a -> In sh Int -> Out sh a -> ())
  TWhere    :: TensorOp (In DIM1 Int -> Out DIM1 Int -> ())

  -- operators from Num
  TAdd  :: OneOf TFNum  a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TMul  :: OneOf TFNum  a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TSub  :: OneOf TFNum  a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TNeg  :: OneOf TFNum' a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TAbs  :: OneOf TFNum' a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TSign :: OneOf TFNum' a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())

  -- operators from Integral
  TTruncateDiv :: OneOf TFNum a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TTruncateMod :: OneOf TFMod a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TRealDiv     :: OneOf TFNum a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())

  -- operators from Bits & FiniteBits
  TBitwiseAnd :: OneOf TFInt a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TBitwiseOr  :: OneOf TFInt a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TBitwiseXor :: OneOf TFInt a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TInvert     :: OneOf TFInt a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())

  -- operators from Fractional and Floating
  TReciprocal :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TSin        :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TCos        :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TTan        :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TAsin       :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TAcos       :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TAtan       :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TSinh       :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TCosh       :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TTanh       :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TAsinh      :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TAcosh      :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TAtanh      :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TSqrt       :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TExp        :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TLog        :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TPow        :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TLog1p      :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TAtan2      :: OneOf TFFloat a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())

  -- relational and equality operators
  TLess         :: OneOf TFOrd a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh PrimBool -> ())
  TGreater      :: OneOf TFOrd a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh PrimBool -> ())
  TLessEqual    :: OneOf TFOrd a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh PrimBool -> ())
  TGreaterEqual :: OneOf TFOrd a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh PrimBool -> ())
  TEqual        :: OneOf TFOrd a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh PrimBool -> ())
  TNotEqual     :: OneOf TFOrd a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh PrimBool -> ())
  TMaximum      :: OneOf TFOrd a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TMinimum      :: OneOf TFOrd a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())

  -- logical operators
  TLogicalAnd :: TensorOp (In sh (PrimBool, PrimBool) -> Out sh PrimBool -> ())
  TLogicalOr  :: TensorOp (In sh (PrimBool, PrimBool) -> Out sh PrimBool -> ())
  TLogicalNot :: TensorOp (In sh PrimBool -> Out sh PrimBool -> ())

  TCast :: (TensorType a, TensorType b) => ScalarType a -> ScalarType b -> TensorOp (In sh a -> Out sh b -> ())

  TTensorScatter :: ScatterFun -> TensorOp (Mut sh' s -> In sh sh' -> In sh s -> ())
  TBooleanMask :: ScalarType s -> TensorOp (In DIM1 s -> In DIM1 PrimBool -> Out DIM1 s -> ())

instance EncodeOperation TensorOp where
  encodeOperation :: TensorOp t -> Builder
  encodeOperation t = case t of
    TConstant st s -> intHost $(hashQ ("Constant" :: String)) <> encodeScalarConst st s
    TVar st -> intHost $(hashQ ("Var" :: String)) <> encodeScalarType st
    TWhere -> intHost $(hashQ ("Where" :: String))
    TId st -> intHost $(hashQ ("Id" :: String)) <> encodeScalarType st
    TSelect st -> intHost $(hashQ ("Select" :: String)) <> encodeScalarType st
    TAdd st -> intHost $(hashQ ("Add" :: String)) <> encodeScalarType st
    TMul st -> intHost $(hashQ ("Mul" :: String)) <> encodeScalarType st
    TSub st -> intHost $(hashQ ("Sub" :: String)) <> encodeScalarType st
    TNeg st -> intHost $(hashQ ("Neg" :: String)) <> encodeScalarType st
    TAbs st -> intHost $(hashQ ("Abs" :: String)) <> encodeScalarType st
    TSign st -> intHost $(hashQ ("Sign" :: String)) <> encodeScalarType st
    TTruncateDiv st -> intHost $(hashQ ("TruncateDiv" :: String)) <> encodeScalarType st
    TTruncateMod st -> intHost $(hashQ ("TruncateMod" :: String)) <> encodeScalarType st
    TTensorScatter sf -> encodeScatterFun sf
    TBooleanMask st -> intHost $(hashQ ("BooleanMask" :: String)) <> encodeScalarType st
    TReciprocal st -> intHost $(hashQ ("Reciprocal" :: String)) <> encodeScalarType st
    TSin st -> intHost $(hashQ ("Sin" :: String)) <> encodeScalarType st
    TCos st -> intHost $(hashQ ("Cos" :: String)) <> encodeScalarType st
    TTan st -> intHost $(hashQ ("Tan" :: String)) <> encodeScalarType st
    TAsin st -> intHost $(hashQ ("Asin" :: String)) <> encodeScalarType st
    TAcos st -> intHost $(hashQ ("Acos" :: String)) <> encodeScalarType st
    TAtan st -> intHost $(hashQ ("Atan" :: String)) <> encodeScalarType st
    TSinh st -> intHost $(hashQ ("Sinh" :: String)) <> encodeScalarType st
    TCosh st -> intHost $(hashQ ("Cosh" :: String)) <> encodeScalarType st
    TTanh st -> intHost $(hashQ ("Tanh" :: String)) <> encodeScalarType st
    TAsinh st -> intHost $(hashQ ("Asinh" :: String)) <> encodeScalarType st
    TAcosh st -> intHost $(hashQ ("Acosh" :: String)) <> encodeScalarType st
    TAtanh st -> intHost $(hashQ ("Atanh" :: String)) <> encodeScalarType st
    TSqrt st -> intHost $(hashQ ("Sqrt" :: String)) <> encodeScalarType st
    TExp st -> intHost $(hashQ ("Exp" :: String)) <> encodeScalarType st
    TLog st -> intHost $(hashQ ("Log" :: String)) <> encodeScalarType st
    TPow st -> intHost $(hashQ ("Pow" :: String)) <> encodeScalarType st
    TLog1p st -> intHost $(hashQ ("Log1p" :: String)) <> encodeScalarType st
    TAtan2 st -> intHost $(hashQ ("Atan2" :: String)) <> encodeScalarType st
    TLess st -> intHost $(hashQ ("Less" :: String)) <> encodeScalarType st
    TGreater st -> intHost $(hashQ ("Greater" :: String)) <> encodeScalarType st
    TLessEqual st -> intHost $(hashQ ("LessEqual" :: String)) <> encodeScalarType st
    TGreaterEqual st -> intHost $(hashQ ("GreaterEqual" :: String)) <> encodeScalarType st
    TEqual st -> intHost $(hashQ ("Equal" :: String)) <> encodeScalarType st
    TNotEqual st -> intHost $(hashQ ("NotEqual" :: String)) <> encodeScalarType st
    TMaximum st -> intHost $(hashQ ("Maximum" :: String)) <> encodeScalarType st
    TMinimum st -> intHost $(hashQ ("Minimum" :: String)) <> encodeScalarType st
    TLogicalAnd -> intHost $(hashQ ("LogicalAnd" :: String))
    TLogicalOr -> intHost $(hashQ ("LogicalOr" :: String))
    TLogicalNot -> intHost $(hashQ ("LogicalNot" :: String))
    TRealDiv st -> intHost $(hashQ ("RealDiv" :: String)) <> encodeScalarType st
    TBitwiseAnd st -> intHost $(hashQ ("BitwiseAnd" :: String)) <> encodeScalarType st
    TBitwiseOr st -> intHost $(hashQ ("BitwiseOr" :: String)) <> encodeScalarType st
    TBitwiseXor st -> intHost $(hashQ ("BitwiseXor" :: String)) <> encodeScalarType st
    TInvert st -> intHost $(hashQ ("Invert" :: String)) <> encodeScalarType st
    TCast st1 st2 -> intHost $(hashQ ("Cast" :: String)) <> encodeScalarType st1 <> encodeScalarType st2
    TGather st -> intHost $(hashQ ("Gather" :: String)) <> encodeScalarType st

instance PrettyOp TensorOp where
  prettyOp :: TensorOp t -> Adoc
  prettyOp (TConstant s e)    = vsep ["TConst"]
  prettyOp (TId s)            = vsep ["TId", prettyScalarType s]
  prettyOp (TTensorScatter f) = vsep ["TTensorScatter", viaShow f]
  prettyOp (TBooleanMask s)   = vsep ["TBooleanMask", prettyScalarType s]
  prettyOp _                  = vsep ["pretty me!"]

instance NFData' TensorOp where
  rnf' !_ = ()

instance SimplifyOperation TensorOp where

instance SLVOperation TensorOp where
  slvOperation _ = Nothing

instance ShrinkArg (BackendClusterArg TensorOp) where
  shrinkArg _ NoFusionArg = NoFusionArg
  deadArg NoFusionArg = NoFusionArg

instance NFData' (BackendClusterArg TensorOp) where
  rnf' NoFusionArg = ()

instance MakesILP TensorOp where
  type BackendVar TensorOp = ()
  type BackendArg TensorOp = ()
  data BackendClusterArg TensorOp arg = NoFusionArg
  mkGraph _ _ _ = mempty
  labelLabelledArg _ _ (L arg l) = LOp arg l ()
  getClusterArg (LOp _ _ ()) = NoFusionArg
  finalize = mempty
  encodeBackendClusterArg NoFusionArg = intHost $(hashQ ("NoFusionArg" :: String))

data ScatterFun where
  ScatterFunAdd :: ScatterFun
  ScatterFunMin :: ScatterFun
  deriving Show

encodeScatterFun :: ScatterFun -> Builder
encodeScatterFun ScatterFunAdd = intHost $(hashQ ("ScatterFunAdd" :: String))
encodeScatterFun ScatterFunMin = intHost $(hashQ ("ScatterFunMin" :: String))

prettyScatterFun :: ScatterFun -> Adoc
prettyScatterFun ScatterFunAdd = "ScatterFunAdd"
prettyScatterFun ScatterFunMin = "ScatterFunMin"
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE TypeOperators     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module Data.Accelerate.TensorFlow.Kernel where
import Data.Accelerate.TensorFlow.Type
    ( TFOrd, OneOf, TFNum, TFNum', TFAll, TFFloat, TensorType, TFInt, TFMod )
import Data.Array.Accelerate.AST.Schedule.Uniform
    ( PrimBool,
      Var(Var),
      PreArgs((:>:)),
      GroundVars,
      Arg(..),
      NFData'(..),
      Args,
      BaseVars,
      BaseVar,
      Cluster,
      BaseR(BaseRground),
      AccessGroundR, GroundR (GroundRscalar), GroundVar )
import Data.Array.Accelerate.Array.Buffer ( Buffer, Buffers )
import Data.Array.Accelerate.Representation.Type
    ( Distributes(reprIsSingle),
      TupR(TupRpair, TupRunit, TupRsingle),
      TypeR, mapTupR )
import Data.Array.Accelerate.Backend
    ( IsKernel(KernelMetadata, compileKernel, KernelOperation),
      PrettyKernel(..),
      PrettyKernelStyle(PrettyKernelBody) )
import Data.Accelerate.TensorFlow.Operation ( TensorOp(..), ScatterFun (..) )
import Data.Array.Accelerate.AST.Kernel ( NoKernelMetadata )
import Data.Array.Accelerate.AST.Environment ( Env )
import Data.Array.Accelerate.Eval
    ( clusterOperations,
      ApplyOperation(ApplyOperation),
      ClusterOperations(ClusterOperations) )
import Data.Array.Accelerate.Analysis.Match ( type (:~:)(..) )
import Data.Array.Accelerate.AST.LeftHandSide
    ( LeftHandSide(LeftHandSidePair, LeftHandSideWildcard) )
import Data.Array.Accelerate.Representation.Array
    ( ArrayR(ArrayR) )
import Data.Array.Accelerate.Type ( ScalarType )
import Data.Array.Accelerate.Representation.Shape (ShapeR, DIM1)

data TensorArg env sh a where
  TensorArg :: ShapeR sh -> BaseVars env sh -> ScalarType a -> BaseVar env (Buffer a) -> TensorArg env sh a

data TensorKernel env where
  TensorConstant    :: TensorType a => TensorArg env sh a -> a -> TensorKernel env
  TensorVar         :: TensorType a => TensorArg env sh a -> BaseVar env a -> TensorKernel env
  TensorId          :: OneOf TFAll a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorSelect      :: OneOf TFAll a => TensorArg env sh PrimBool -> TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorWhere       :: OneOf TFAll a => TensorArg env DIM1 a -> TensorArg env DIM1 Int -> TensorKernel env
  TensorGather      :: OneOf TFAll a => TensorArg env DIM1 a -> TensorArg env sh Int -> TensorArg env sh a -> TensorKernel env
  TensorCast :: (TensorType a, TensorType b) => TensorArg env sh a -> TensorArg env sh b -> TensorKernel env

  -- scatter operations
  TensorScatterAdd  :: TensorType a => TensorArg env DIM1 a -> TensorArg env DIM1 Int -> TensorArg env DIM1 a -> TensorKernel env

  -- operators from Num
  TensorAdd  :: OneOf TFNum  a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorMul  :: OneOf TFNum  a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorSub  :: OneOf TFNum  a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorNeg  :: OneOf TFNum' a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorAbs  :: OneOf TFNum' a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorSign :: OneOf TFNum' a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env

  -- operators from Integral
  TensorTruncateDiv :: OneOf TFNum a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorTruncateMod :: OneOf TFMod a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorRealDiv     :: OneOf TFNum a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env

  -- operators from Bits & FiniteBits
  TensorBitwiseAnd :: OneOf TFInt a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorBitwiseOr  :: OneOf TFInt a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorBitwiseXor :: OneOf TFInt a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorInvert     :: OneOf TFInt a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env

  -- operators from Fractional and Floating
  TensorReciprocal :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorSin        :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorCos        :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorTan        :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorAsin       :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorAcos       :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorAtan       :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorSinh       :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorCosh       :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorTanh       :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorAsinh      :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorAcosh      :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorAtanh      :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorExp        :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorSqrt       :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorLog        :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorPow        :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorLog1p      :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorKernel env

  -- operators from RealFrac
  TensorRound    :: (OneOf TFFloat a, OneOf TFInt b) => TensorArg env sh a -> TensorArg env sh b -> TensorKernel env
  TensorFloor    :: (OneOf TFFloat a, OneOf TFInt b) => TensorArg env sh a -> TensorArg env sh b -> TensorKernel env
  TensorCeil     :: (OneOf TFFloat a, OneOf TFInt b) => TensorArg env sh a -> TensorArg env sh b -> TensorKernel env

  -- operators from RealFloat
  TensorAtan2      :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorIsNan      :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh PrimBool -> TensorKernel env
  TensorIsInf      :: OneOf TFFloat a => TensorArg env sh a -> TensorArg env sh PrimBool -> TensorKernel env

  -- relational and equality operators
  TensorLess         :: OneOf TFOrd a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh PrimBool -> TensorKernel env
  TensorGreater      :: OneOf TFOrd a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh PrimBool -> TensorKernel env
  TensorLessEqual    :: OneOf TFOrd a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh PrimBool -> TensorKernel env
  TensorGreaterEqual :: OneOf TFOrd a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh PrimBool -> TensorKernel env
  TensorEqual        :: OneOf TFOrd a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh PrimBool -> TensorKernel env
  TensorNotEqual     :: OneOf TFOrd a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh PrimBool -> TensorKernel env
  TensorMaximum      :: OneOf TFOrd a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env
  TensorMinimum      :: OneOf TFOrd a => TensorArg env sh a -> TensorArg env sh a -> TensorArg env sh a -> TensorKernel env

  TensorLogicalAnd   :: TensorArg env sh PrimBool -> TensorArg env sh PrimBool -> TensorArg env sh PrimBool -> TensorKernel env
  TensorLogicalOr    :: TensorArg env sh PrimBool -> TensorArg env sh PrimBool -> TensorArg env sh PrimBool -> TensorKernel env
  TensorLogicalNot   :: TensorArg env sh PrimBool -> TensorArg env sh PrimBool -> TensorKernel env

instance NFData' TensorKernel where
  rnf' !_  = ()

newtype TensorFlowKernelMetadata f =
  TensorFlowKernelMetadata { kernelArgsSize :: Int }

instance IsKernel TensorKernel where
  type KernelOperation TensorKernel = TensorOp
  type KernelMetadata  TensorKernel = NoKernelMetadata

  compileKernel :: Env AccessGroundR env -> Cluster TensorOp args -> Args env args -> TensorKernel env
  compileKernel _ cluster clusterArgs
    | ClusterOperations _ lhs [ApplyOperation operation args] <- clusterOperations cluster clusterArgs
    , Just Refl <- wildcards lhs
    = compileOperation operation args
      where
      wildcards :: LeftHandSide a e env env' -> Maybe (env :~: env')
      wildcards (LeftHandSideWildcard _)    = Just Refl
      wildcards (LeftHandSidePair lhs lhs')
        | Just Refl <- wildcards lhs
        , Just Refl <- wildcards lhs'        = Just Refl
      wildcards _                           = Nothing
  compileKernel _ _ _ = error "impossible, did you use SequentialSchedule?"

instance PrettyKernel TensorKernel where
  prettyKernel :: PrettyKernelStyle TensorKernel
  prettyKernel = PrettyKernelBody True $ \_ _ -> ""

compileOperation :: TensorOp args -> Args env args -> TensorKernel env
compileOperation (TConstant _ s) (aOut :>: _)                               = TensorConstant (arg aOut) s
compileOperation (TVar st) (ArgVar (TupRsingle (Var _ idx)) :>: aOut :>: _) = TensorVar (arg aOut) (Var (BaseRground (GroundRscalar st)) idx)
compileOperation (TVar _) _                                                 = error "impossible"
compileOperation TId (aIn :>: aOut :>: _)                                   = TensorId (arg aIn) (arg aOut)
compileOperation TSelect (aIn1 :>: aIn2 :>: aIn3 :>: aOut :>: _)            = TensorSelect (arg aIn1) (arg aIn2) (arg aIn3) (arg aOut)
compileOperation TWhere (aIn :>: aOut :>: _)                                = TensorWhere (arg aIn) (arg aOut)
compileOperation TGather (aIn1 :>: aIn2 :>: aOut :>: _)                     = TensorGather (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TCast (aIn :>: aOut :>: _)                                 = TensorCast (arg aIn) (arg aOut)
compileOperation (TTensorScatter ScatterFunAdd) (aMut :>: aIn1 :>: aIn2 :>: _) = TensorScatterAdd (arg aMut) (arg aIn1) (arg aIn2)

compileOperation TAdd (aIn1 :>: aIn2 :>: aOut :>: _)                        = TensorAdd (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TMul (aIn1 :>: aIn2 :>: aOut :>: _)                        = TensorMul (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TSub (aIn1 :>: aIn2 :>: aOut :>: _)                        = TensorSub (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TNeg (aIn :>: aOut :>: _)                                  = TensorNeg (arg aIn) (arg aOut)
compileOperation TAbs (aIn :>: aOut :>: _)                                  = TensorAbs (arg aIn) (arg aOut)
compileOperation TSign (aIn :>: aOut :>: _)                                 = TensorSign (arg aIn) (arg aOut)

compileOperation TTruncateDiv (aIn1 :>: aIn2 :>: aOut :>: _)                = TensorTruncateDiv (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TTruncateMod (aIn1 :>: aIn2 :>: aOut :>: _)                = TensorTruncateMod (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TRealDiv (aIn1 :>: aIn2 :>: aOut :>: _)                    = TensorRealDiv (arg aIn1) (arg aIn2) (arg aOut)

compileOperation TBitwiseAnd (aIn1 :>: aIn2 :>: aOut :>: _)                 = TensorBitwiseAnd (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TBitwiseOr (aIn1 :>: aIn2 :>: aOut :>: _)                  = TensorBitwiseOr (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TBitwiseXor (aIn1 :>: aIn2 :>: aOut :>: _)                 = TensorBitwiseXor (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TInvert (aIn :>: aOut :>: _)                               = TensorInvert (arg aIn) (arg aOut)

compileOperation TReciprocal (aIn :>: aOut :>: _)                           = TensorReciprocal (arg aIn) (arg aOut)
compileOperation TSin (aIn :>: aOut :>: _)                                  = TensorSin (arg aIn) (arg aOut)
compileOperation TCos (aIn :>: aOut :>: _)                                  = TensorCos (arg aIn) (arg aOut)
compileOperation TTan (aIn :>: aOut :>: _)                                  = TensorTan (arg aIn) (arg aOut)
compileOperation TAsin (aIn :>: aOut :>: _)                                 = TensorAsin (arg aIn) (arg aOut)
compileOperation TAcos (aIn :>: aOut :>: _)                                 = TensorAcos (arg aIn) (arg aOut)
compileOperation TAtan (aIn :>: aOut :>: _)                                 = TensorAtan (arg aIn) (arg aOut)
compileOperation TSinh (aIn :>: aOut :>: _)                                 = TensorSinh (arg aIn) (arg aOut)
compileOperation TCosh (aIn :>: aOut :>: _)                                 = TensorCosh (arg aIn) (arg aOut)
compileOperation TTanh (aIn :>: aOut :>: _)                                 = TensorTanh (arg aIn) (arg aOut)
compileOperation TAsinh (aIn :>: aOut :>: _)                                = TensorAsinh (arg aIn) (arg aOut)
compileOperation TAcosh (aIn :>: aOut :>: _)                                = TensorAcosh (arg aIn) (arg aOut)
compileOperation TAtanh (aIn :>: aOut :>: _)                                = TensorAtanh (arg aIn) (arg aOut)
compileOperation TSqrt (aIn :>: aOut :>: _)                                 = TensorSqrt (arg aIn) (arg aOut)
compileOperation TExp (aIn :>: aOut :>: _)                                  = TensorExp (arg aIn) (arg aOut)
compileOperation TLog (aIn :>: aOut :>: _)                                  = TensorLog (arg aIn) (arg aOut)
compileOperation TPow (aIn1 :>: aIn2 :>: aOut :>: _)                        = TensorPow (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TLog1p (aIn :>: aOut :>: _)                                = TensorLog1p (arg aIn) (arg aOut)
compileOperation TAtan2 (aIn1 :>: aIn2 :>: aOut :>: _)                      = TensorAtan2 (arg aIn1) (arg aIn2) (arg aOut)

compileOperation TRound (aIn :>: aOut :>: _)                                = TensorRound (arg aIn) (arg aOut)
compileOperation TFloor (aIn :>: aOut :>: _)                                = TensorFloor (arg aIn) (arg aOut)
compileOperation TCeil (aIn :>: aOut :>: _)                                 = TensorCeil (arg aIn) (arg aOut)

compileOperation TIsNan (aIn :>: aOut :>: _)                                = TensorIsNan (arg aIn) (arg aOut)
compileOperation TIsInf (aIn :>: aOut :>: _)                                = TensorIsInf (arg aIn) (arg aOut)

compileOperation TLess (aIn1 :>: aIn2 :>: aOut :>: _)                       = TensorLess (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TGreater (aIn1 :>: aIn2 :>: aOut :>: _)                    = TensorGreater (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TLessEqual (aIn1 :>: aIn2 :>: aOut :>: _)                  = TensorLessEqual (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TGreaterEqual (aIn1 :>: aIn2 :>: aOut :>: _)               = TensorGreaterEqual (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TEqual (aIn1 :>: aIn2 :>: aOut :>: _)                      = TensorEqual (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TNotEqual (aIn1 :>: aIn2 :>: aOut :>: _)                   = TensorNotEqual (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TMaximum (aIn1 :>: aIn2 :>: aOut :>: _)                    = TensorMaximum (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TMinimum (aIn1 :>: aIn2 :>: aOut :>: _)                    = TensorMinimum (arg aIn1) (arg aIn2) (arg aOut)

compileOperation TLogicalAnd (aIn1 :>: aIn2 :>: aOut :>: _)                 = TensorLogicalAnd (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TLogicalOr (aIn1 :>: aIn2 :>: aOut :>: _)                  = TensorLogicalOr (arg aIn1) (arg aIn2) (arg aOut)
compileOperation TLogicalNot (aIn :>: aOut :>: _)                           = TensorLogicalNot (arg aIn) (arg aOut)

arg :: forall env a m sh. Arg env (m sh a) -> TensorArg env sh a
arg (ArgArray _ (ArrayR sh a@(TupRsingle st)) gv gvb) 
  | Refl <- reprIsSingle @ScalarType @a @Buffer st
  = TensorArg sh (groundsToBase' gv) st (groundToBase a gvb)
arg _ = error "impossible"

groundToBase :: TypeR a -> GroundVars env (Buffer a) -> BaseVar env (Buffer a)
groundToBase _ (TupRsingle (Var groundR idx)) = Var (BaseRground groundR) idx

groundToBase' :: GroundVar env a -> BaseVar env a
groundToBase' (Var groundR idx) = Var (BaseRground groundR) idx

groundsToBase :: TypeR a -> GroundVars env (Buffers a) -> BaseVars env (Buffers a)
groundsToBase _ TupRunit                              = TupRunit
groundsToBase t@(TupRsingle (st :: ScalarType e)) gvb
  | Refl <- reprIsSingle @ScalarType @e @Buffer st    = TupRsingle (groundToBase t gvb)
groundsToBase (TupRpair t1 t2) (TupRpair gvb1 gvb2)   = TupRpair (groundsToBase t1 gvb1) (groundsToBase t2 gvb2)
groundsToBase _ _                                     = error "impossible"

groundsToBase' :: GroundVars env a -> BaseVars env a
groundsToBase' = mapTupR groundToBase'                